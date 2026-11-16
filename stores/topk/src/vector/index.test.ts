import { MastraError } from '@mastra/core/error';
import { field } from 'topk-js/query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('topk-js/query', async importOriginal => {
  const actual = await importOriginal<typeof import('topk-js/query')>();
  return {
    ...actual,
    select: vi.fn(actual.select),
    match: vi.fn(actual.match),
    should: vi.fn(actual.should),
    fn: {
      ...actual.fn,
      semanticSimilarity: vi.fn(actual.fn.semanticSimilarity),
      bm25Score: vi.fn(actual.fn.bm25Score),
      vectorDistance: vi.fn(actual.fn.vectorDistance),
      multiVectorDistance: vi.fn(actual.fn.multiVectorDistance),
    },
  };
});

const queryModule = await import('topk-js/query');
const { TopKVector } = await import('./index');

type Doc = Record<string, any>;

/** In-memory stand-in for the parts of the topk-js client the store uses. Records every call. */
class FakeTopK {
  schemas = new Map<string, Record<string, any>>();
  docs = new Map<string, Map<string, Doc>>();
  calls: { op: string; collection?: string; partition?: string; args: any[] }[] = [];
  queryRows: Doc[] = [];
  lsn = 0;

  collections() {
    return {
      create: async (name: string, schema: Record<string, any>) => {
        this.calls.push({ op: 'create', collection: name, args: [schema] });
        if (this.schemas.has(name)) throw new Error('collection already exists');
        this.schemas.set(name, schemaToSpec(schema));
        return { name, schema: this.schemas.get(name) };
      },
      get: async (name: string) => {
        if (!this.schemas.has(name)) throw new Error('collection not found');
        return { name, schema: this.schemas.get(name) };
      },
      delete: async (name: string) => {
        this.calls.push({ op: 'deleteCollection', collection: name, args: [] });
        if (!this.schemas.delete(name)) throw new Error('collection not found');
      },
      list: (options?: { prefix?: string }) =>
        (async function* (schemas: Map<string, any>) {
          for (const name of schemas.keys()) {
            if (!options?.prefix || name.startsWith(options.prefix)) yield { name };
          }
        })(this.schemas),
    };
  }

  collection(name: string, partition?: string) {
    const key = `${name}/${partition ?? ''}`;
    const record = (op: string, ...args: any[]) => this.calls.push({ op, collection: name, partition, args });
    const store = () => {
      if (!this.docs.has(key)) this.docs.set(key, new Map());
      return this.docs.get(key)!;
    };
    return {
      upsert: async (rows: Doc[]) => {
        record('upsert', rows);
        for (const row of rows) store().set(row._id, row);
        return String(++this.lsn);
      },
      query: async (q: unknown, options?: unknown) => {
        record('query', q, options);
        if (partition === 'missing') throw new Error('partition not found');
        return this.queryRows;
      },
      get: async (ids: string[], fields: string[], options?: unknown) => {
        record('get', ids, fields, options);
        const out: Record<string, Doc> = {};
        for (const id of ids) {
          const doc = store().get(id);
          if (doc) out[id] = Object.fromEntries(fields.filter(f => f in doc).map(f => [f, doc[f]]));
        }
        return out;
      },
      delete: async (target: unknown) => {
        record('delete', target);
        return String(++this.lsn);
      },
      count: async (options?: unknown) => {
        record('count', options);
        if (partition === 'missing') throw new Error('partition not found');
        return store().size;
      },
      listPartitions: () =>
        (async function* () {
          yield { name: 'acme' };
        })(),
      deletePartition: async (p: string) => record('deletePartition', p),
    };
  }

  lastCall(op: string) {
    return [...this.calls].reverse().find(c => c.op === op);
  }
}

/** Mirrors the shape `collections().get()` returns, from the FieldSpec objects passed to create(). */
function schemaToSpec(schema: Record<string, any>): Record<string, any> {
  const out: Record<string, any> = {};
  for (const [name, spec] of Object.entries(schema)) {
    const s = String(spec);
    const dim = /dimension: (\d+)/.exec(s)?.[1];
    const dataType = s.includes('F32SparseVector')
      ? { type: 'F32SparseVector' }
      : s.includes('Matrix')
        ? { type: 'Matrix', dimension: Number(dim), valueType: 'f32' }
        : dim
          ? { type: 'F32Vector', dimension: Number(dim) }
          : { type: 'Text' };
    let index: any;
    if (s.includes('SemanticIndex')) index = { type: 'SemanticIndex' };
    else if (s.includes('KeywordIndex')) index = { type: 'KeywordIndex', indexType: 'text' };
    else if (s.includes('MultiVectorIndex')) index = { type: 'MultiVectorIndex', metric: 'maxsim' };
    else if (s.includes('VectorIndex')) {
      const metric = /metric: (\w+)/.exec(s)?.[1] ?? 'Cosine';
      index = {
        type: 'VectorIndex',
        metric: { Cosine: 'cosine', Euclidean: 'euclidean', DotProduct: 'dot_product' }[metric] ?? metric.toLowerCase(),
      };
    }
    out[name] = { dataType, required: false, ...(index ? { index } : {}) };
  }
  return out;
}

let fake: FakeTopK;
let store: InstanceType<typeof TopKVector>;

beforeEach(() => {
  vi.clearAllMocks();
  fake = new FakeTopK();
  store = new TopKVector({ id: 'topk-test', client: fake as any });
});

describe('TopKVector', () => {
  describe('client', () => {
    it('is self-embedding, so Memory needs no embedder', () => {
      expect(store.isSelfEmbedding).toBe(true);
    });

    it('explains how to set missing credentials', async () => {
      const previous = { key: process.env.TOPK_API_KEY, region: process.env.TOPK_REGION };
      delete process.env.TOPK_API_KEY;
      delete process.env.TOPK_REGION;
      try {
        const bare = new TopKVector({ id: 'bare' });
        await expect(bare.listIndexes()).rejects.toThrow(/TOPK_API_KEY/);
        const keyOnly = new TopKVector({ id: 'key-only', apiKey: 'k' });
        await expect(keyOnly.listIndexes()).rejects.toThrow(/TOPK_REGION/);
      } finally {
        if (previous.key) process.env.TOPK_API_KEY = previous.key;
        if (previous.region) process.env.TOPK_REGION = previous.region;
      }
    });
  });

  describe('createIndex', () => {
    it('creates a semantic collection when no dimension is given', async () => {
      await store.createIndex({ indexName: 'docs' });
      const schema = fake.lastCall('create')!.args[0];
      expect(Object.keys(schema)).toEqual(['text']);
      expect(String(schema.text)).toContain('SemanticIndex');
    });

    it('creates a vector field and a BM25 text index with a dimension', async () => {
      await store.createIndex({ indexName: 'docs', dimension: 3, metric: 'dotproduct' });
      const schema = fake.lastCall('create')!.args[0];
      expect(String(schema.text)).toContain('KeywordIndex');
      expect(String(schema.vector)).toMatch(/dimension: 3/);
      expect(String(schema.vector)).toMatch(/DotProduct/);
    });

    it('adds sparse and multi-vector fields', async () => {
      await store.createIndex({ indexName: 'docs', textIndex: 'none', sparse: true, multiVector: { dimension: 8 } });
      const schema = fake.lastCall('create')!.args[0];
      expect(Object.keys(schema).sort()).toEqual(['multi_vector', 'sparse_vector', 'text']);
      expect(String(schema.multi_vector)).toContain('MultiVectorIndex');
    });

    it('accepts an existing compatible index without recreating it', async () => {
      await store.createIndex({ indexName: 'docs', dimension: 3 });
      await store.createIndex({ indexName: 'docs', dimension: 3 });
      expect(fake.calls.filter(c => c.op === 'create')).toHaveLength(1);
    });

    it('rejects a dimension that differs from the existing index', async () => {
      await store.createIndex({ indexName: 'docs', dimension: 3 });
      await expect(store.createIndex({ indexName: 'docs', dimension: 4 })).rejects.toThrow(/3 dimensions/);
    });

    it('validates its arguments', async () => {
      await expect(store.createIndex({ indexName: 'x', dimension: 0 })).rejects.toThrow(MastraError);
      await expect(store.createIndex({ indexName: 'x', dimension: 3, metric: 'manhattan' as any })).rejects.toThrow(
        /metric/,
      );
      await expect(store.createIndex({ indexName: 'x', metric: 'cosine' })).rejects.toThrow(/needs a dimension/);
      await expect(store.createIndex({ indexName: 'x', textIndex: 'none' })).rejects.toThrow(/nothing to search/);
      await expect(store.createIndex({ indexName: '' })).rejects.toThrow(/indexName/);
    });

    it('uses the index prefix for collection names and listIndexes', async () => {
      const prefixed = new TopKVector({ id: 'p', client: fake as any, indexPrefix: 'staging-' });
      await prefixed.createIndex({ indexName: 'docs' });
      await store.createIndex({ indexName: 'other' });
      expect(fake.schemas.has('staging-docs')).toBe(true);
      expect(await prefixed.listIndexes()).toEqual(['docs']);
    });
  });

  describe('upsert', () => {
    beforeEach(async () => {
      await store.createIndex({ indexName: 'docs', dimension: 2 });
    });

    it('writes text, vector and three copies of the metadata', async () => {
      const [id] = await store.upsert({
        indexName: 'docs',
        ids: ['a'],
        vectors: [[0.1, 0.2]],
        metadata: [{ text: 'hello', tags: ['x', 'y'], scores: [1, 2.5], nested: { n: 1 }, flags: [true] }],
      });
      expect(id).toBe('a');
      const [row] = fake.lastCall('upsert')!.args[0];
      expect(row._id).toBe('a');
      expect(row.text).toBe('hello');
      expect(row.vector).toEqual([0.1, 0.2]);
      expect(JSON.parse(row.metadata_json)).toEqual({
        text: 'hello',
        tags: ['x', 'y'],
        scores: [1, 2.5],
        nested: { n: 1 },
        flags: [true],
      });
      expect(String(row.metadata_keys)).toContain('nested.n');
      expect(row.metadata_sizes).toEqual({ tags: 2, scores: 2, flags: 1 });
      expect(String(row.metadata.tags)).toContain('String');
      expect(String(row.metadata.scores)).toContain('F64');
      expect(row.metadata.flags).toBeUndefined();
      expect(row.metadata.nested).toEqual({ n: 1 });
    });

    it('generates ids when none are given', async () => {
      const ids = await store.upsert({ indexName: 'docs', vectors: [[1, 0]] });
      expect(ids[0]).toMatch(/^[0-9a-f-]{36}$/);
    });

    it('validates lengths, values and dimensions', async () => {
      await expect(store.upsert({ indexName: 'docs', vectors: [] })).rejects.toThrow(MastraError);
      await expect(store.upsert({ indexName: 'docs', vectors: [[1, 0]], ids: ['a', 'b'] })).rejects.toThrow(/ids/);
      await expect(store.upsert({ indexName: 'docs', vectors: [[1, 0]], metadata: [{}, {}] })).rejects.toThrow(
        /metadata/,
      );
      await expect(store.upsert({ indexName: 'docs', vectors: [[Number.NaN, 0]] })).rejects.toThrow(MastraError);
      await expect(store.upsert({ indexName: 'docs', vectors: [[1, 0, 0]] })).rejects.toThrow(/3 dimensions/);
      await expect(store.upsert({ indexName: 'missing', vectors: [[1, 0]] })).rejects.toThrow(/not found/);
    });

    it('rejects metadata that is not JSON-serializable', async () => {
      const circular: any = {};
      circular.self = circular;
      await expect(store.upsert({ indexName: 'docs', vectors: [[1, 0]], metadata: [circular] })).rejects.toThrow(
        /JSON-serializable/,
      );
    });

    it('deletes matching documents before writing when deleteFilter is given', async () => {
      await store.upsert({ indexName: 'docs', vectors: [[1, 0]], deleteFilter: { source: 'a.pdf' } });
      const ops = fake.calls.map(c => c.op);
      expect(ops.indexOf('delete')).toBeLessThan(ops.lastIndexOf('upsert'));
    });

    it('splits large writes into several requests', async () => {
      await store.upsert({ indexName: 'docs', vectors: Array.from({ length: 1200 }, () => [1, 0]) });
      expect(fake.calls.filter(c => c.op === 'upsert').length).toBeGreaterThan(1);
    });
  });

  describe('upsert into a semantic collection', () => {
    beforeEach(async () => {
      await store.createIndex({ indexName: 'faq' });
    });

    it('stores documents for TopK to embed', async () => {
      await store.upsert({ indexName: 'faq', ids: ['1'], documents: ['Refunds take 14 days.'] });
      expect(fake.lastCall('upsert')!.args[0][0].text).toBe('Refunds take 14 days.');
    });

    it('rejects vectors and text a semantic index cannot hold', async () => {
      await expect(store.upsert({ indexName: 'faq', vectors: [[1]] })).rejects.toThrow(/pass documents/);
      await expect(store.upsert({ indexName: 'faq', documents: ['x'.repeat(4097)] })).rejects.toThrow(/maxChunkChars/);
      await expect(store.upsert({ indexName: 'faq', documents: [''], ids: ['a', 'b'] })).rejects.toThrow(/ids/);
    });
  });

  describe('query', () => {
    beforeEach(async () => {
      await store.createIndex({ indexName: 'faq' });
      await store.createIndex({ indexName: 'dense', dimension: 2 });
      await store.createIndex({ indexName: 'euclid', dimension: 2, metric: 'euclidean' });
      fake.queryRows = [{ _id: 'a', score: 4, text: 'hello', metadata_json: '{"source":"x"}' }];
    });

    it('runs semantic search for queryText alone', async () => {
      const results = await store.query({ indexName: 'faq', queryText: 'hi', topK: 3 });
      expect(queryModule.fn.semanticSimilarity).toHaveBeenCalledWith('text', 'hi');
      expect(queryModule.fn.bm25Score).toHaveBeenCalled();
      expect(queryModule.should).toHaveBeenCalledWith('hi', { field: 'text' });
      expect(results).toEqual([{ id: 'a', score: 4, document: 'hello', metadata: { source: 'x' } }]);
    });

    it('runs keyword search with a required match', async () => {
      await store.query({ indexName: 'faq', queryText: 'W-200', mode: 'keyword', keywordMatchAll: true });
      expect(queryModule.match).toHaveBeenCalledWith('W-200', { field: 'text', all: true });
      expect(queryModule.fn.semanticSimilarity).not.toHaveBeenCalled();
    });

    it('uses only the signals with a non-zero weight', async () => {
      await store.query({ indexName: 'faq', queryText: 'hi', weights: { semantic: 1, keyword: 0 } });
      expect(queryModule.fn.semanticSimilarity).toHaveBeenCalled();
      expect(queryModule.fn.bm25Score).not.toHaveBeenCalled();
    });

    it('rejects weights for signals the query has no input for', async () => {
      await expect(store.query({ indexName: 'faq', queryText: 'hi', weights: { vector: 1 } })).rejects.toThrow(
        /pass queryVector/,
      );
      await expect(store.query({ indexName: 'faq', queryText: 'hi', weights: { bogus: 1 } as any })).rejects.toThrow(
        /Unknown hybrid weight/,
      );
    });

    it('runs vector search for queryVector alone', async () => {
      await store.query({ indexName: 'dense', queryVector: [1, 0] });
      expect(queryModule.fn.vectorDistance).toHaveBeenCalledWith('vector', [1, 0]);
      expect(queryModule.fn.bm25Score).not.toHaveBeenCalled();
    });

    it('turns squared euclidean distance into a higher-is-better score', async () => {
      fake.queryRows = [{ _id: 'a', score: 9, metadata_json: '{}' }];
      const [result] = await store.query({ indexName: 'euclid', queryVector: [1, 0] });
      expect(result!.score).toBeCloseTo(0.25);
    });

    it('passes a custom scoring expression', async () => {
      const scoring = vi.fn((s: any) => s.semantic.add(s.keyword));
      await store.query({ indexName: 'faq', queryText: 'hi', scoring });
      expect(scoring).toHaveBeenCalledWith(
        expect.objectContaining({ semantic: expect.anything(), keyword: expect.anything() }),
      );
      await expect(store.query({ indexName: 'faq', queryText: 'hi', scoring: () => 1 as any })).rejects.toThrow(
        /TopK expression/,
      );
    });

    it('lists documents for a filter-only query', async () => {
      fake.queryRows = [{ _id: 'a', metadata_json: '{}' }];
      const results = await store.query({ indexName: 'dense', filter: { a: 1 } });
      expect(results).toEqual([{ id: 'a', score: 0, metadata: {} }]);
      const selected = (queryModule.select as any).mock.calls.at(-1)[0];
      expect(Object.keys(selected)).toEqual(['text', 'metadata_json']);
    });

    it('reads vectors and missing metadata with get()', async () => {
      await store.upsert({ indexName: 'dense', ids: ['a'], vectors: [[1, 0]], metadata: [{ k: 1 }] });
      fake.queryRows = [{ _id: 'a', score: 1 }];
      const [result] = await store.query({ indexName: 'dense', queryVector: [1, 0], includeVector: true });
      expect(result!.vector).toEqual([1, 0]);
      expect(result!.metadata).toEqual({ k: 1 });
    });

    it('validates topK, vectors and dimensions', async () => {
      await expect(store.query({ indexName: 'dense', queryVector: [1, 0], topK: 0 })).rejects.toThrow(MastraError);
      await expect(store.query({ indexName: 'dense', queryVector: [1, 0], topK: 1.5 })).rejects.toThrow(MastraError);
      await expect(store.query({ indexName: 'dense', queryVector: [1, 0], topK: 10_001 })).rejects.toThrow(
        /at most 10000/,
      );
      await expect(store.query({ indexName: 'dense', queryVector: [1, 0, 0] })).rejects.toThrow(/3 dimensions/);
      await expect(store.query({ indexName: 'faq', queryVector: [1, 0] })).rejects.toThrow(/no dense vector field/);
      await expect(store.query({ indexName: 'missing', queryText: 'x' })).rejects.toThrow(/not found/);
    });

    it('returns no results for a partition that was never written', async () => {
      expect(await store.query({ indexName: 'faq', queryText: 'x', partition: 'missing' })).toEqual([]);
    });

    it('passes a raw TopK filter expression through', async () => {
      await store.query({ indexName: 'dense', queryVector: [1, 0], filter: field('metadata.year').gte(2020) });
      expect(fake.lastCall('query')).toBeDefined();
    });
  });

  describe('consistency and partitions', () => {
    beforeEach(async () => {
      await store.createIndex({ indexName: 'faq' });
      fake.queryRows = [];
    });

    it('reads its own last write by default', async () => {
      await store.upsert({ indexName: 'faq', documents: ['a'] });
      await store.query({ indexName: 'faq', queryText: 'a' });
      expect(fake.lastCall('query')!.args[1]).toEqual({ lsn: store.lastLsn('faq') });
    });

    it('prefers an explicit lsn, then an explicit consistency', async () => {
      await store.upsert({ indexName: 'faq', documents: ['a'] });
      await store.query({ indexName: 'faq', queryText: 'a', lsn: '42' });
      expect(fake.lastCall('query')!.args[1]).toEqual({ lsn: '42' });
      await store.query({ indexName: 'faq', queryText: 'a', consistency: 'strong' });
      expect(fake.lastCall('query')!.args[1]).toEqual({ consistency: 'strong' });
    });

    it('uses the store default consistency when read-your-writes is off', async () => {
      const relaxed = new TopKVector({ id: 'r', client: fake as any, consistency: 'indexed', readYourWrites: false });
      await relaxed.upsert({ indexName: 'faq', documents: ['a'] });
      await relaxed.query({ indexName: 'faq', queryText: 'a' });
      expect(fake.lastCall('query')!.args[1]).toEqual({ consistency: 'indexed' });
    });

    it('routes reads and writes to a partition', async () => {
      const acme = store.withPartition('acme');
      await acme.upsert({ indexName: 'faq', documents: ['a'] });
      expect(fake.lastCall('upsert')!.partition).toBe('acme');
      await store.query({ indexName: 'faq', queryText: 'a', partition: 'globex' });
      expect(fake.lastCall('query')!.partition).toBe('globex');
      expect(acme.lastLsn('faq')).toBeDefined();
      expect(store.lastLsn('faq')).toBeUndefined();
      expect(await store.listPartitions({ indexName: 'faq' })).toEqual(['acme']);
      await store.deletePartition({ indexName: 'faq', partition: 'acme' });
      expect(fake.lastCall('deletePartition')!.args).toEqual(['acme']);
    });

    it('counts zero documents in a partition that was never written', async () => {
      expect(await store.describeIndex({ indexName: 'faq', partition: 'missing' })).toEqual({ dimension: 0, count: 0 });
    });
  });

  describe('updateVector', () => {
    beforeEach(async () => {
      await store.createIndex({ indexName: 'dense', dimension: 2 });
      await store.upsert({
        indexName: 'dense',
        ids: ['a'],
        vectors: [[1, 0]],
        metadata: [{ text: 'old text', keep: false, gone: 1 }],
      });
    });

    it('replaces metadata and keeps the vector', async () => {
      await store.updateVector({ indexName: 'dense', id: 'a', update: { metadata: { text: 'new text', keep: true } } });
      const [row] = fake.lastCall('upsert')!.args[0];
      expect(JSON.parse(row.metadata_json)).toEqual({ text: 'new text', keep: true });
      expect(row.vector).toEqual([1, 0]);
      // The text came from metadata.text, so it follows the new value.
      expect(row.text).toBe('new text');
    });

    it('replaces the vector and keeps the metadata', async () => {
      await store.updateVector({ indexName: 'dense', id: 'a', update: { vector: [0, 1] } });
      const [row] = fake.lastCall('upsert')!.args[0];
      expect(row.vector).toEqual([0, 1]);
      expect(JSON.parse(row.metadata_json)).toEqual({ text: 'old text', keep: false, gone: 1 });
    });

    it('updates every document matching a filter', async () => {
      fake.queryRows = [{ _id: 'a' }];
      await store.updateVector({ indexName: 'dense', filter: { keep: false }, update: { metadata: { keep: true } } });
      expect(JSON.parse(fake.lastCall('upsert')!.args[0][0].metadata_json)).toEqual({ keep: true });
    });

    it('validates its arguments', async () => {
      await expect(store.updateVector({ indexName: 'dense', update: { vector: [1, 0] } } as any)).rejects.toThrow(
        /id or filter/,
      );
      await expect(
        store.updateVector({ indexName: 'dense', id: 'a', filter: { a: 1 }, update: { vector: [1, 0] } } as any),
      ).rejects.toThrow(/mutually exclusive/);
      await expect(store.updateVector({ indexName: 'dense', id: 'a', update: {} })).rejects.toThrow(/No updates/);
      await expect(
        store.updateVector({ indexName: 'dense', filter: {}, update: { metadata: { a: 1 } } } as any),
      ).rejects.toThrow(/empty filter/);
      await expect(store.updateVector({ indexName: 'dense', id: 'a', update: { vector: [1, 0, 0] } })).rejects.toThrow(
        /3 dimensions/,
      );
    });
  });

  describe('deletes', () => {
    beforeEach(async () => {
      await store.createIndex({ indexName: 'faq' });
    });

    it('deletes by id and by filter natively', async () => {
      await store.deleteVector({ indexName: 'faq', id: 'a' });
      expect(fake.lastCall('delete')!.args[0]).toEqual(['a']);
      await store.deleteVectors({ indexName: 'faq', ids: ['a', 'b'] });
      expect(fake.lastCall('delete')!.args[0]).toEqual(['a', 'b']);
      await store.deleteVectors({ indexName: 'faq', filter: { source: 'x' } });
      expect(String(fake.lastCall('delete')!.args[0])).toContain('metadata.source');
    });

    it('validates its arguments', async () => {
      await expect(store.deleteVectors({ indexName: 'faq', ids: [] })).rejects.toThrow(/empty ids array/);
      await expect(store.deleteVectors({ indexName: 'faq', filter: {} })).rejects.toThrow(/empty filter/);
      await expect(store.deleteVectors({ indexName: 'faq' })).rejects.toThrow(/filter or ids/);
      await expect(store.deleteVectors({ indexName: 'faq', ids: ['a'], filter: { a: 1 } })).rejects.toThrow(
        /mutually exclusive/,
      );
      await expect(store.deleteVectors({ indexName: 'missing', ids: ['a'] })).rejects.toThrow(/not found/);
    });

    it('deletes an index idempotently', async () => {
      await store.deleteIndex({ indexName: 'faq' });
      await store.deleteIndex({ indexName: 'faq' });
      await expect(store.describeIndex({ indexName: 'faq' })).rejects.toThrow(/not found/);
    });
  });
});
