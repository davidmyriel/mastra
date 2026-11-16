import { createVectorTestSuite } from '@internal/storage-test-utils';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { TopKVector, metadataField } from './index';

/**
 * Live tests against TopK. They run only when TOPK_API_KEY is set (TOPK_REGION defaults to
 * aws-us-east-1-elastica). Every collection gets a prefix unique to this run, and cleanup deletes
 * only collections with that prefix, so the tests never touch anything else in the project.
 */
const apiKey = process.env.TOPK_API_KEY;
const region = process.env.TOPK_REGION ?? 'aws-us-east-1-elastica';
const describeLive = apiKey ? describe : describe.skip;

const runPrefix = `mastra-it-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}-`;

function makeStore(suffix: string) {
  return new TopKVector({ id: `topk-it-${suffix}`, apiKey, region, indexPrefix: `${runPrefix}${suffix}-` });
}

async function cleanup(store: TopKVector) {
  for (const indexName of await store.listIndexes()) {
    await store.deleteIndex({ indexName }).catch(() => {});
  }
}

describeLive('TopKVector Integration Tests', () => {
  describe('Mastra shared vector suite', () => {
    const store = makeStore('suite');
    afterAll(() => cleanup(store), 300_000);

    createVectorTestSuite({
      vector: store,
      createIndex: async (indexName, options) => {
        await store.createIndex({ indexName, dimension: 1536, metric: options?.metric });
      },
      deleteIndex: async indexName => {
        await store.deleteIndex({ indexName });
      },
      // Reads pass the LSN of this store's last write, so they always see it.
      waitForIndexing: async () => {},
      supportsElemMatch: false,
    });
  });

  describe('server-side embeddings', () => {
    const store = makeStore('semantic');
    const indexName = 'faq';

    beforeAll(async () => {
      await store.createIndex({ indexName });
      await store.upsert({
        indexName,
        ids: ['refunds', 'shipping', 'returns', 'warranty'],
        documents: [
          'Refunds are issued to the original payment method within 14 days.',
          'Standard shipping takes 3-5 business days. Express shipping takes 1-2 days.',
          'You can return unused items within 30 days for a full refund.',
          'Every product has a two-year warranty against manufacturing defects (error code W-200).',
        ],
        metadata: [
          { source: 'billing.md', year: 2024 },
          { source: 'shipping.md', year: 2023 },
          { source: 'billing.md', year: 2022 },
          { source: 'warranty.md', year: 2024 },
        ],
      });
    }, 120_000);
    afterAll(() => cleanup(store), 120_000);

    it('reports itself as self-embedding', () => {
      expect(store.isSelfEmbedding).toBe(true);
    });

    it('runs semantic search alone for queryText, without BM25', async () => {
      const [semantic] = await store.query({ indexName, queryText: 'warranty', mode: 'semantic', topK: 1 });
      const [inferred] = await store.query({ indexName, queryText: 'warranty', topK: 1 });
      expect(inferred?.score).toBeCloseTo(semantic!.score, 5);
    });

    it('describes a collection TopK embeds itself', async () => {
      const stats = await store.describeIndex({ indexName });
      expect(stats).toEqual({ dimension: 0, count: 4 });
    });

    it('answers a natural-language query with no embedding model', async () => {
      const results = await store.query({ indexName, queryText: 'how long until I get my money back?', topK: 2 });
      expect(results[0]?.id).toBe('refunds');
      expect(results[0]?.document).toContain('Refunds');
      expect(results[0]?.metadata).toEqual({ source: 'billing.md', year: 2024 });
      expect(results[0]!.score).toBeGreaterThan(results[1]!.score);
    });

    it('filters semantic results by metadata', async () => {
      const results = await store.query({
        indexName,
        queryText: 'money back',
        filter: { source: 'billing.md', year: { $lt: 2024 } },
      });
      expect(results.map(r => r.id)).toEqual(['returns']);
    });

    it('runs BM25 keyword search', async () => {
      const results = await store.query({ indexName, queryText: 'W-200', mode: 'keyword' });
      expect(results.map(r => r.id)).toEqual(['warranty']);
    });

    it('combines semantic and keyword scores in one hybrid query', async () => {
      const results = await store.query({
        indexName,
        queryText: 'express delivery',
        mode: 'hybrid',
        weights: { semantic: 0.7, keyword: 0.3 },
        topK: 4,
      });
      expect(results).toHaveLength(4);
      expect(results[0]?.id).toBe('shipping');
    });

    it('ranks by a custom scoring expression', async () => {
      const results = await store.query({
        indexName,
        queryText: 'refund',
        scoring: s => s.semantic!.add(s.keyword!).boost(metadataField('source').eq('warranty.md'), 100),
        topK: 4,
      });
      expect(results[0]?.id).toBe('warranty');
    });

    it('lists matching documents with only a filter', async () => {
      const results = await store.query({ indexName, filter: { source: 'billing.md' }, topK: 10 });
      expect(results.map(r => r.id).sort()).toEqual(['refunds', 'returns']);
      expect(results.every(r => r.score === 0)).toBe(true);
    });

    it('updates a document text and re-embeds it', async () => {
      await store.updateVector({ indexName, id: 'returns', update: { document: 'Our office cat is named Pixel.' } });
      const results = await store.query({ indexName, queryText: 'what is the name of the cat?', topK: 1 });
      expect(results[0]?.id).toBe('returns');
      expect(results[0]?.metadata).toEqual({ source: 'billing.md', year: 2022 });
    });

    it('embeds long text, such as a default-size Memory chunk', async () => {
      const filler = 'The weather report mentions clouds and light wind over the hills. ';
      await store.upsert({
        indexName,
        ids: ['long'],
        documents: [filler.repeat(240) + 'The launch code for project Falcon is BLUE-77.'],
        metadata: [{ source: 'notes.md' }],
      });
      const results = await store.query({ indexName, queryText: 'what is the Falcon launch code?', topK: 1 });
      expect(results[0]?.id).toBe('long');
      await store.deleteVector({ indexName, id: 'long' });
    });

    it('explains text that is too long for a semantic index', async () => {
      await expect(store.upsert({ indexName, documents: ['x'.repeat(40_000)] })).rejects.toThrow(/split long text/);
    });

    it('rejects vectors in a collection TopK embeds itself', async () => {
      await expect(store.upsert({ indexName, vectors: [[0.1, 0.2]] })).rejects.toThrow(/pass documents/);
    });
  });

  describe('hybrid search over client-side vectors', () => {
    const store = makeStore('dense');
    const indexName = 'chunks';

    beforeAll(async () => {
      await store.createIndex({ indexName, dimension: 3, metric: 'cosine' });
      // Mastra RAG chunks keep their text in metadata.text, which is indexed for BM25.
      await store.upsert({
        indexName,
        ids: ['a', 'b', 'c'],
        vectors: [
          [1, 0, 0],
          [0, 1, 0],
          [0.7, 0.7, 0],
        ],
        metadata: [
          { text: 'Rust guarantees memory safety without a garbage collector', lang: 'rust' },
          { text: 'Python has readable syntax', lang: 'python' },
          { text: 'Go has a garbage collector and goroutines', lang: 'go' },
        ],
      });
    }, 120_000);
    afterAll(() => cleanup(store), 120_000);

    it('defaults to a keyword index for collections with a dimension', async () => {
      const results = await store.query({ indexName, queryText: 'goroutines', mode: 'keyword' });
      expect(results.map(r => r.id)).toEqual(['c']);
    });

    it('mixes vector similarity and BM25', async () => {
      const vectorOnly = await store.query({ indexName, queryVector: [0, 1, 0], topK: 1 });
      expect(vectorOnly[0]?.id).toBe('b');

      const hybrid = await store.query({
        indexName,
        queryVector: [0, 1, 0],
        queryText: 'garbage collector',
        weights: { vector: 0.2, keyword: 1 },
        topK: 1,
      });
      expect(hybrid[0]?.id).not.toBe('b');
    });

    it('returns vectors when asked', async () => {
      const [result] = await store.query({ indexName, queryVector: [1, 0, 0], topK: 1, includeVector: true });
      expect(result?.vector).toEqual([1, 0, 0]);
      expect(result!.score).toBeCloseTo(1, 5);
    });
  });

  describe('euclidean scores', () => {
    const store = makeStore('euclid');
    afterAll(() => cleanup(store), 120_000);

    it('turns squared distances into higher-is-better scores', async () => {
      await store.createIndex({ indexName: 'points', dimension: 2, metric: 'euclidean' });
      await store.upsert({
        indexName: 'points',
        ids: ['near', 'far'],
        vectors: [
          [1, 0],
          [4, 4],
        ],
      });
      const results = await store.query({ indexName: 'points', queryVector: [1, 0] });
      expect(results.map(r => r.id)).toEqual(['near', 'far']);
      expect(results[0]!.score).toBeCloseTo(1, 5);
      expect(results[1]!.score).toBeCloseTo(1 / (1 + 5), 5);
      expect((await store.describeIndex({ indexName: 'points' })).metric).toBe('euclidean');
    });
  });

  describe('sparse and multi-vector search', () => {
    const store = makeStore('sparse');
    const indexName = 'mixed';

    beforeAll(async () => {
      await store.createIndex({ indexName, textIndex: 'keyword', sparse: true, multiVector: { dimension: 2 } });
      await store.upsert({
        indexName,
        ids: ['a', 'b'],
        documents: ['alpha', 'beta'],
        sparseVectors: [
          { indices: [1, 7], values: [0.5, 1] },
          { indices: [2], values: [1] },
        ],
        multiVectors: [
          [
            [1, 0],
            [0, 1],
          ],
          [[0.6, 0.8]],
        ],
      });
    }, 120_000);
    afterAll(() => cleanup(store), 120_000);

    it('searches sparse vectors', async () => {
      const results = await store.query({ indexName, sparseVector: { indices: [7], values: [1] } });
      expect(results[0]?.id).toBe('a');
      expect(results[0]!.score).toBeCloseTo(1, 5);
    });

    it('searches multi-vectors with MaxSim', async () => {
      const results = await store.query({ indexName, multiVector: [[0.6, 0.8]] });
      expect(results[0]?.id).toBe('b');
    });

    it('keeps sparse and multi-vectors when metadata is updated', async () => {
      await store.updateVector({ indexName, id: 'a', update: { metadata: { tag: 'updated' } } });
      const [result] = await store.query({ indexName, sparseVector: { indices: [7], values: [1] }, topK: 1 });
      expect(result?.id).toBe('a');
      expect(result?.metadata).toEqual({ tag: 'updated' });
    });
  });

  describe('partitions and consistency', () => {
    const store = makeStore('tenants');
    const indexName = 'memory';

    beforeAll(async () => {
      await store.createIndex({ indexName });
      await store.withPartition('acme').upsert({ indexName, ids: ['1'], documents: ['Acme sells rocket skates.'] });
      await store.upsert({
        indexName,
        partition: 'globex',
        ids: ['1'],
        documents: ['Globex builds doomsday devices.'],
      });
    }, 120_000);
    afterAll(() => cleanup(store), 120_000);

    it('keeps each partition isolated', async () => {
      const acme = await store.query({ indexName, partition: 'acme', queryText: 'what does the company make?' });
      expect(acme.map(r => r.document)).toEqual(['Acme sells rocket skates.']);
      const globex = await store.withPartition('globex').query({ indexName, queryText: 'what does the company make?' });
      expect(globex.map(r => r.document)).toEqual(['Globex builds doomsday devices.']);
      expect(await store.query({ indexName, queryText: 'company' })).toEqual([]);
    });

    it('treats a partition that was never written as empty', async () => {
      expect(await store.query({ indexName, partition: 'initech', queryText: 'anything' })).toEqual([]);
      expect((await store.describeIndex({ indexName, partition: 'initech' })).count).toBe(0);
    });

    it('lists partitions', async () => {
      expect((await store.listPartitions({ indexName })).sort()).toEqual(['acme', 'globex']);
    });

    it('reads at an LSN or with strong consistency', async () => {
      await store.upsert({ indexName, partition: 'acme', ids: ['2'], documents: ['Acme also sells anvils.'] });
      const lsn = store.lastLsn(indexName, 'acme');
      expect(lsn).toBeTruthy();
      const atLsn = await store.query({ indexName, partition: 'acme', queryText: 'anvils', lsn, topK: 1 });
      expect(atLsn[0]?.id).toBe('2');
      const strong = await store.query({
        indexName,
        partition: 'acme',
        queryText: 'anvils',
        consistency: 'strong',
        topK: 1,
      });
      expect(strong[0]?.id).toBe('2');
      const indexed = await store.query({ indexName, partition: 'acme', queryText: 'anvils', consistency: 'indexed' });
      expect(Array.isArray(indexed)).toBe(true);
    });
  });
});
