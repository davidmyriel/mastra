---
'@mastra/topk': minor
---

Added `@mastra/topk`, a vector store backed by [TopK](https://topk.io). TopK embeds text on its servers, so semantic search and Memory semantic recall work without an embedding model or key, and one query can combine semantic, BM25 keyword and vector scores with metadata filters.

```ts
import { TopKVector } from '@mastra/topk';

const topk = new TopKVector({ id: 'topk' }); // reads TOPK_API_KEY and TOPK_REGION

await topk.createIndex({ indexName: 'faq' });
await topk.upsert({ indexName: 'faq', documents: ['Refunds are issued within 14 days.'] });

const results = await topk.query({
  indexName: 'faq',
  queryText: 'how long do refunds take?',
  mode: 'hybrid',
  weights: { semantic: 0.7, keyword: 0.3 },
});
```

- Pass `new TopKVector()` to `Memory` with no `embedder` for semantic recall embedded by TopK.
- Indexes with a `dimension` take your own embeddings and work with `createVectorQueryTool`.
- Search modes: semantic, BM25 keyword, vector, sparse, multi-vector (MaxSim) and hybrid, plus custom scoring expressions.
- Partitions per tenant (`withPartition()`), and read consistency per store or per query.
