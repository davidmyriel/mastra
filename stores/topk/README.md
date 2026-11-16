# @mastra/topk

[TopK](https://topk.io) vector store for Mastra. TopK is a search database with built-in semantic search: it embeds and reranks text on its servers, so semantic search and Memory semantic recall need no embedding model or key. One query can combine semantic, BM25 keyword, dense, sparse and multi-vector scores with metadata filters.

## Installation

```bash
npm install @mastra/topk
```

Create an API key in the [TopK console](https://console.topk.io) and set your [region](https://docs.topk.io/regions):

```bash
export TOPK_API_KEY="your-api-key"
export TOPK_REGION="aws-us-east-1-elastica"
```

## Usage

```typescript
import { TopKVector } from '@mastra/topk';

const topk = new TopKVector({ id: 'topk' });

// No dimension: TopK embeds the text itself
await topk.createIndex({ indexName: 'faq' });
await topk.upsert({
  indexName: 'faq',
  documents: ['Refunds are issued within 14 days.', 'Express shipping takes 1-2 days.'],
  metadata: [{ source: 'billing.md' }, { source: 'shipping.md' }],
});

// Semantic search, no embedding model needed
await topk.query({ indexName: 'faq', queryText: 'how long do refunds take?', topK: 3 });

// BM25 keyword search
await topk.query({ indexName: 'faq', queryText: 'express', mode: 'keyword' });

// Semantic + keyword + metadata filter in one query
await topk.query({
  indexName: 'faq',
  queryText: 'refund policy',
  mode: 'hybrid',
  weights: { semantic: 0.7, keyword: 0.3 },
  filter: { source: { $in: ['billing.md'] } },
});
```

Your own embeddings work too: `createIndex({ indexName, dimension: 1536 })`, then `upsert({ vectors })` and `query({ queryVector })`. Text in `metadata.text`, as Mastra's RAG chunks have, is indexed for BM25, so the same index answers keyword and hybrid queries.

### Memory semantic recall without an embedder

```typescript
import { Memory } from '@mastra/memory';
import { TopKVector } from '@mastra/topk';

const memory = new Memory({
  storage,
  vector: new TopKVector({ id: 'topk' }),
  options: { semanticRecall: { topK: 3, messageRange: 2, maxChunkChars: 4000 } },
});
```

A semantic index holds at most 4,096 characters per document, so keep `maxChunkChars` at 4,000 or less.

### More

- Custom scoring with any TopK expression: `scoring: s => s.semantic!.add(s.keyword!).boost(metadataField('featured').eq(true), 1.5)`
- Sparse vectors (`createIndex({ sparse: true })`) and multi-vector MaxSim search (`createIndex({ multiVector: { dimension } })`)
- Every Mastra filter operator except `$elemMatch`, or a raw `topk-js` filter expression
- Partitions per tenant: `topk.withPartition('acme')`, or `partition` on any call
- Read consistency per store or per query: `consistency: 'indexed' | 'strong'`, or `lsn`. Reads see the store's own writes by default.

See the [TopK vector store documentation](https://mastra.ai/integrations/vectors/topk) and the [TopK documentation](https://docs.topk.io).

## Development

```bash
pnpm test   # unit tests; with TOPK_API_KEY set, also the live tests and Mastra's shared vector suite
```

Live tests create collections named `mastra-it-<run>-...` and delete only those.
