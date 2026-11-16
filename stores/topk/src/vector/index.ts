import { ErrorCategory, ErrorDomain, MastraError } from '@mastra/core/error';
import { createVectorErrorId } from '@mastra/core/storage';
import type {
  CreateIndexParams,
  DeleteIndexParams,
  DeleteVectorParams,
  DeleteVectorsParams,
  DescribeIndexParams,
  IndexStats,
  QueryResult,
  QueryVectorParams,
  SparseVector,
  UpdateVectorParams,
  UpsertVectorParams,
} from '@mastra/core/vector';
import { MastraVector, validateTopK, validateVectorValues } from '@mastra/core/vector';
import { Client } from 'topk-js';
import type { ClientConfig, CollectionFieldSpec, ConsistencyLevel, QueryOptions, query, RetryConfig } from 'topk-js';
import { f32SparseVector, f64List, i64List, matrix, stringList } from 'topk-js/data';
import { field, fn, match, select, should } from 'topk-js/query';
import {
  f32SparseVector as sparseVectorSpec,
  f32Vector,
  keywordIndex,
  matrix as matrixSpec,
  multiVectorIndex,
  semanticIndex,
  text,
  vectorIndex,
} from 'topk-js/schema';
import { TOPK_FIELDS, TopKFilterTranslator } from './filter';
import type { TopKVectorFilter } from './filter';

export { TOPK_FIELDS, TopKFilterTranslator, metadataField } from './filter';
export type { TopKVectorFilter } from './filter';

const STORE = 'TOPK';

/** TopK returns at most this many results from one query. */
export const TOPK_MAX_RESULTS = 10_000;

/** The longest text a `semantic` index accepts in one document. */
export const TOPK_MAX_SEMANTIC_TEXT_LENGTH = 4_096;

/** Target size of one upsert request, well under TopK's 8MB limit. */
const MAX_REQUEST_BYTES = 4 * 1024 * 1024;

/** The largest dense vector dimension TopK accepts. */
const MAX_DIMENSION = 16_384;

type Metric = 'cosine' | 'euclidean' | 'dotproduct';
const TOPK_METRICS: Record<Metric, 'cosine' | 'euclidean' | 'dot_product'> = {
  cosine: 'cosine',
  euclidean: 'euclidean',
  dotproduct: 'dot_product',
};

/**
 * Index created on the `text` field of a new collection.
 * - `semantic`: TopK embeds and reranks the text on the server, and also indexes it for BM25.
 *   At most 4,096 characters per document.
 * - `keyword`: BM25 keyword search only, with no length limit beyond the document size.
 * - `none`: the text is stored and returned, but isn't searchable.
 */
export type TopKTextIndex = 'semantic' | 'keyword' | 'none';

/**
 * Read consistency. Leave it unset for TopK's default, balanced consistency (writes are
 * visible in under a second). `indexed` reads only fully indexed data, for the lowest latency.
 * `strong` always sees the latest writes, at a higher cost.
 * See https://docs.topk.io/concepts#read-consistency
 */
export type TopKConsistency = ConsistencyLevel;

/** The scores a query can combine. */
export type TopKSignal = 'semantic' | 'keyword' | 'vector' | 'sparse' | 'multiVector';

/**
 * How a query ranks documents.
 * - `semantic`: TopK's server-side embeddings and reranking of `queryText`. No embedding model needed.
 * - `keyword`: BM25 over `queryText` (or `keywordQuery`). Only documents containing a query term are returned.
 * - `vector`: dense vector similarity to `queryVector`.
 * - `sparse`: sparse vector similarity to `sparseVector`.
 * - `multiVector`: late-interaction (MaxSim) similarity to `multiVector`.
 * - `hybrid`: a weighted sum of every signal the query provides input for, in one query.
 */
export type TopKSearchMode = TopKSignal | 'hybrid';

/** Weight of each signal in a hybrid query. Signals left out aren't used. */
export type TopKHybridWeights = Partial<Record<TopKSignal, number>>;

/**
 * Score expressions passed to a custom `scoring` function, one per signal the query provides input
 * for. They're plain TopK expressions, so they combine with `add`, `mul`, `boost` and the rest. The
 * keyword score is 0 for documents without a query term.
 */
export type TopKScoreSignals = Partial<Record<TopKSignal, query.LogicalExpression>>;

export interface TopKVectorConfig {
  /** Unique identifier for this vector store instance. */
  id: string;
  /** TopK API key. Defaults to the `TOPK_API_KEY` environment variable. Create one at https://console.topk.io */
  apiKey?: string;
  /** TopK region, for example `aws-us-east-1-elastica`. Defaults to `TOPK_REGION`. See https://docs.topk.io/regions */
  region?: string;
  /** Custom TopK host. Leave unset for TopK's cloud. */
  host?: string;
  /** Whether to connect over HTTPS. Defaults to `true`. */
  https?: boolean;
  /** Retry settings passed to the TopK client. */
  retryConfig?: RetryConfig;
  /** An existing `topk-js` client to use instead of `apiKey` and `region`. */
  client?: Client;
  /**
   * Prefix added to every index name to form the TopK collection name. `listIndexes()` only
   * returns collections with this prefix, without it. Use it to share one TopK project between
   * apps or environments, for example `indexPrefix: 'staging-'`.
   */
  indexPrefix?: string;
  /** Partition (tenant) used by every call that doesn't pass its own `partition`. */
  partition?: string;
  /** Default read consistency. Leave unset for TopK's balanced default. */
  consistency?: TopKConsistency;
  /**
   * Make reads see this store's own latest write to the same index and partition, by passing
   * that write's LSN. Applies to reads that set neither `lsn` nor `consistency`. Defaults to `true`.
   */
  readYourWrites?: boolean;
}

/** Selects a TopK partition (tenant). Partitions are isolated: a query never returns another partition's documents. */
export interface TopKPartitionParam {
  /** Partition to read or write. Defaults to the store's `partition`. */
  partition?: string;
}

/** Read consistency for one call. */
export interface TopKReadParams extends TopKPartitionParam {
  /** Consistency for this read. Overrides the store's default. */
  consistency?: TopKConsistency;
  /** Read at least up to this LSN (returned by `lastLsn()` after a write). Takes precedence over `consistency`. */
  lsn?: string;
}

export interface TopKCreateIndexParams extends Omit<CreateIndexParams, 'dimension'> {
  /**
   * Dense vector dimension, for vectors you embed yourself. Leave it out to create a collection
   * that TopK embeds on the server: pass `documents` to `upsert()` and `queryText` to `query()`.
   */
  dimension?: number;
  /** Index for the `text` field. Defaults to `semantic` without a `dimension`, and `keyword` (BM25) with one. */
  textIndex?: TopKTextIndex;
  /** Add a sparse vector field (for example SPLADE), searched exactly with 100% recall. */
  sparse?: boolean;
  /** Add a multi-vector field for late-interaction (ColBERT-style) search scored with MaxSim. */
  multiVector?: { dimension: number };
}

export interface TopKUpsertVectorParams
  extends Omit<UpsertVectorParams<TopKVectorFilter>, 'vectors'>, TopKPartitionParam {
  /** Dense vectors. Required unless the collection has no dense vector field. */
  vectors?: number[][];
  /**
   * Text of each document, stored in the `text` field. In a `semantic` collection TopK embeds it on
   * the server. When left out, a string `metadata.text` is used, as Mastra's RAG chunks provide,
   * so BM25 and hybrid search work on an existing pipeline.
   */
  documents?: string[];
  /** Multi-vector matrices (one row per token), for collections created with `multiVector`. */
  multiVectors?: number[][][];
}

export interface TopKQueryVectorParams extends QueryVectorParams<TopKVectorFilter>, TopKReadParams {
  /** Natural-language query. Used for semantic search (embedded on the server) and for BM25. */
  queryText?: string;
  /** BM25 query, when it should differ from `queryText`. */
  keywordQuery?: string;
  /** In `keyword` mode, require every query term instead of any. */
  keywordMatchAll?: boolean;
  /** Query matrix for multi-vector search. */
  multiVector?: number[][];
  /** How to rank documents. Inferred from the inputs when left out; see {@link TopKSearchMode}. */
  mode?: TopKSearchMode;
  /** Hybrid weights, for example `{ semantic: 0.7, keyword: 0.3 }`. Defaults to 1 for each signal with input. */
  weights?: TopKHybridWeights;
  /**
   * Custom scoring expression. Receives the score expression of each signal the query provides
   * input for and returns the expression to rank by (descending). Use any TopK expression, for
   * example a boost from metadata: `s => s.semantic!.add(s.keyword!).boost(metadataField('featured').eq(true), 1.5)`.
   */
  scoring?: (signals: TopKScoreSignals) => query.FunctionExpression | query.LogicalExpression;
}

export type TopKUpdateVectorParams = UpdateVectorParams<TopKVectorFilter> &
  TopKPartitionParam & {
    update: { vector?: number[]; metadata?: Record<string, any>; document?: string };
  };

export interface TopKDeleteVectorParams extends DeleteVectorParams, TopKPartitionParam {}
export interface TopKDeleteVectorsParams extends DeleteVectorsParams<TopKVectorFilter>, TopKPartitionParam {}
export interface TopKDescribeIndexParams extends DescribeIndexParams, TopKReadParams {}

/** What a collection holds, read from its schema. */
interface CollectionInfo {
  textIndex: TopKTextIndex;
  dimension?: number;
  metric?: Metric;
  sparse: boolean;
  multiVectorDimension?: number;
}

interface SharedState {
  client?: Client;
  infos: Map<string, CollectionInfo>;
  lsns: Map<string, string>;
}

type Row = Record<string, unknown>;

/**
 * Mastra vector store backed by [TopK](https://topk.io): a search database with built-in semantic
 * search (embedded and reranked on the server), BM25 keyword search, dense, sparse and multi-vector
 * search, metadata filters and custom scoring, all combinable in one query.
 *
 * Each Mastra index is a TopK collection. Documents written by this store hold the text, the
 * vectors, the metadata as a struct (for filters) and the metadata as JSON (for results).
 */
export class TopKVector extends MastraVector<TopKVectorFilter> {
  private readonly config: Omit<TopKVectorConfig, 'client'>;
  private readonly shared: SharedState;
  private readonly translator = new TopKFilterTranslator();
  /** Partition this instance reads and writes by default. */
  readonly partition: string | undefined;

  constructor(config: TopKVectorConfig) {
    super({ id: config.id });
    const { client, ...rest } = config;
    this.config = rest;
    this.partition = config.partition;
    this.shared = { client, infos: new Map(), lsns: new Map() };
  }

  /**
   * TopK embeds text on the server, so Mastra Memory can use this store for semantic recall
   * without an `embedder`. When Memory has an embedder, it sends vectors as usual.
   */
  override get isSelfEmbedding(): boolean {
    return true;
  }

  /**
   * Returns a store scoped to another partition (tenant). It shares this store's client, schema
   * cache and settings, and makes no network calls.
   *
   * @example
   * ```ts
   * const acme = vector.withPartition('acme');
   * await acme.upsert({ indexName: 'docs', documents: ['...'] });
   * ```
   */
  withPartition(partition: string | undefined): TopKVector {
    const scoped = Object.create(Object.getPrototypeOf(this)) as TopKVector;
    Object.assign(scoped, this, { partition });
    return scoped;
  }

  /** The underlying `topk-js` client. */
  get client(): Client {
    if (!this.shared.client) {
      const apiKey = this.config.apiKey ?? process.env.TOPK_API_KEY;
      const region = this.config.region ?? process.env.TOPK_REGION;
      if (!apiKey || !region) {
        throw new MastraError({
          id: createVectorErrorId(STORE, 'CLIENT', 'MISSING_CREDENTIALS'),
          domain: ErrorDomain.MASTRA_VECTOR,
          category: ErrorCategory.USER,
          text: !apiKey
            ? 'TopK API key not found. Pass `apiKey` or set TOPK_API_KEY. Create a key at https://console.topk.io'
            : 'TopK region not found. Pass `region` or set TOPK_REGION, for example "aws-us-east-1-elastica". See https://docs.topk.io/regions',
        });
      }
      const clientConfig: ClientConfig = { apiKey, region };
      if (this.config.host !== undefined) clientConfig.host = this.config.host;
      if (this.config.https !== undefined) clientConfig.https = this.config.https;
      if (this.config.retryConfig !== undefined) clientConfig.retryConfig = this.config.retryConfig;
      this.shared.client = new Client(clientConfig);
    }
    return this.shared.client;
  }

  /** LSN of this store's latest write to an index and partition. Pass it as `lsn` to read that write from anywhere. */
  lastLsn(indexName: string, partition?: string): string | undefined {
    return this.shared.lsns.get(this.lsnKey(indexName, partition ?? this.partition));
  }

  // ---------------------------------------------------------------------------
  // Indexes
  // ---------------------------------------------------------------------------

  async createIndex(params: TopKCreateIndexParams): Promise<void> {
    const { indexName, dimension, metric, sparse = false, multiVector } = params;
    const fail = (status: string, text: string, details: Record<string, any> = {}) =>
      new MastraError({
        id: createVectorErrorId(STORE, 'CREATE_INDEX', status),
        domain: ErrorDomain.MASTRA_VECTOR,
        category: ErrorCategory.USER,
        text,
        details: { indexName, ...details },
      });

    this.assertIndexName(indexName, 'CREATE_INDEX');
    if (dimension !== undefined && (!Number.isInteger(dimension) || dimension <= 0 || dimension > MAX_DIMENSION)) {
      throw fail('INVALID_DIMENSION', `dimension must be an integer between 1 and ${MAX_DIMENSION}`, { dimension });
    }
    if (metric !== undefined && !(metric in TOPK_METRICS)) {
      throw fail('INVALID_METRIC', `metric must be "cosine", "euclidean" or "dotproduct", got "${String(metric)}"`);
    }
    if (metric !== undefined && dimension === undefined) {
      throw fail('METRIC_WITHOUT_DIMENSION', 'metric applies to dense vectors, so it needs a dimension');
    }
    const textIndex = params.textIndex ?? (dimension === undefined ? 'semantic' : 'keyword');
    if (!['semantic', 'keyword', 'none'].includes(textIndex)) {
      throw fail('INVALID_TEXT_INDEX', `textIndex must be "semantic", "keyword" or "none", got "${String(textIndex)}"`);
    }
    if (
      multiVector !== undefined &&
      (!Number.isInteger(multiVector.dimension) || multiVector.dimension <= 0 || multiVector.dimension > 1024)
    ) {
      throw fail('INVALID_MULTI_VECTOR_DIMENSION', 'multiVector.dimension must be an integer between 1 and 1024');
    }
    if (dimension === undefined && textIndex === 'none' && !sparse && !multiVector) {
      throw fail(
        'NOTHING_TO_SEARCH',
        'The index would have nothing to search: pass a dimension, a text index, sparse or multiVector',
      );
    }

    const requested: CollectionInfo = {
      textIndex,
      dimension,
      metric: dimension === undefined ? undefined : (metric ?? 'cosine'),
      sparse,
      multiVectorDimension: multiVector?.dimension,
    };

    try {
      const existing = await this.loadInfo(indexName);
      if (existing) {
        this.checkCompatible(indexName, existing, requested, params);
        return;
      }
      const schema: Record<string, any> = {};
      if (textIndex === 'semantic') schema[TOPK_FIELDS.text] = text().index(semanticIndex());
      else if (textIndex === 'keyword') schema[TOPK_FIELDS.text] = text().index(keywordIndex());
      else schema[TOPK_FIELDS.text] = text();
      if (dimension !== undefined) {
        schema[TOPK_FIELDS.vector] = f32Vector({ dimension }).index(
          vectorIndex({ metric: TOPK_METRICS[requested.metric!] }),
        );
      }
      if (sparse) {
        schema[TOPK_FIELDS.sparseVector] = sparseVectorSpec().index(vectorIndex({ metric: 'dot_product' }));
      }
      if (multiVector) {
        schema[TOPK_FIELDS.multiVector] = matrixSpec({ dimension: multiVector.dimension, valueType: 'f32' }).index(
          multiVectorIndex({ metric: 'maxsim' }),
        );
      }
      try {
        await this.client.collections().create(this.collectionName(indexName), schema);
        this.shared.infos.set(indexName, requested);
      } catch (error) {
        if (!hasMessage(error, 'collection already exists')) throw error;
        // Created concurrently: check it like any existing collection.
        this.shared.infos.delete(indexName);
        const raced = await this.loadInfo(indexName);
        if (raced) this.checkCompatible(indexName, raced, requested, params);
      }
    } catch (error) {
      throw this.wrap(error, 'CREATE_INDEX', { indexName });
    }
  }

  async listIndexes(): Promise<string[]> {
    try {
      const prefix = this.config.indexPrefix ?? '';
      const names: string[] = [];
      for await (const collection of this.client.collections().list(prefix ? { prefix } : undefined)) {
        if (collection.name.startsWith(prefix)) names.push(collection.name.slice(prefix.length));
      }
      return names;
    } catch (error) {
      throw this.wrap(error, 'LIST_INDEXES');
    }
  }

  /**
   * Returns the dimension, metric and document count of an index. The count is for one partition.
   * For a collection TopK embeds itself (created without a dimension), `dimension` is 0.
   */
  async describeIndex(params: TopKDescribeIndexParams): Promise<IndexStats> {
    const { indexName } = params;
    try {
      const info = await this.requireInfo(indexName, 'DESCRIBE_INDEX');
      let count = 0;
      try {
        count = await this.collection(indexName, params.partition).count(this.readOptions(indexName, params));
      } catch (error) {
        if (!hasMessage(error, 'partition not found')) throw error;
      }
      return { dimension: info.dimension ?? 0, count, ...(info.metric ? { metric: info.metric } : {}) };
    } catch (error) {
      throw this.wrap(error, 'DESCRIBE_INDEX', { indexName });
    }
  }

  /**
   * Deletes the TopK collection behind an index, with every partition. Deleting an index that
   * doesn't exist succeeds.
   *
   * For about a minute after a delete, writes to a collection recreated under the same name can be
   * lost. Use a new name rather than deleting and recreating one.
   */
  async deleteIndex({ indexName }: DeleteIndexParams): Promise<void> {
    try {
      this.assertIndexName(indexName, 'DELETE_INDEX');
      try {
        await this.client.collections().delete(this.collectionName(indexName));
      } catch (error) {
        if (!hasMessage(error, 'collection not found')) throw error;
      }
      this.forget(indexName);
    } catch (error) {
      throw this.wrap(error, 'DELETE_INDEX', { indexName });
    }
  }

  /** Lists the partitions (tenants) of an index. */
  async listPartitions({ indexName }: { indexName: string }): Promise<string[]> {
    try {
      await this.requireInfo(indexName, 'LIST_PARTITIONS');
      const names: string[] = [];
      for await (const partition of this.client.collection(this.collectionName(indexName)).listPartitions()) {
        names.push(partition.name);
      }
      return names;
    } catch (error) {
      throw this.wrap(error, 'LIST_PARTITIONS', { indexName });
    }
  }

  /**
   * Deletes a partition (tenant) and all of its documents. The deletion takes about a minute to
   * apply, and the same partition name can't be written to again for a while.
   */
  async deletePartition({ indexName, partition }: { indexName: string; partition: string }): Promise<void> {
    try {
      if (!partition) {
        throw new MastraError({
          id: createVectorErrorId(STORE, 'DELETE_PARTITION', 'MISSING_PARTITION'),
          domain: ErrorDomain.MASTRA_VECTOR,
          category: ErrorCategory.USER,
          text: 'partition is required',
          details: { indexName },
        });
      }
      await this.requireInfo(indexName, 'DELETE_PARTITION');
      await this.client.collection(this.collectionName(indexName)).deletePartition(partition);
      this.shared.lsns.delete(this.lsnKey(indexName, partition));
    } catch (error) {
      throw this.wrap(error, 'DELETE_PARTITION', { indexName, partition });
    }
  }

  // ---------------------------------------------------------------------------
  // Writes
  // ---------------------------------------------------------------------------

  /**
   * Inserts or replaces documents. Pass `vectors`, `documents` (text TopK embeds on the server in a
   * `semantic` index), or both, plus optional `sparseVectors` and `multiVectors`.
   *
   * Writes aren't transactional: with `deleteFilter`, matching documents are deleted first and the
   * new ones written after. Retrying with the same `ids` is safe.
   */
  async upsert(params: TopKUpsertVectorParams): Promise<string[]> {
    const { indexName, vectors, documents, sparseVectors, multiVectors, metadata, ids, deleteFilter } = params;
    const userError = (status: string, text: string, details: Record<string, any> = {}) =>
      new MastraError({
        id: createVectorErrorId(STORE, 'UPSERT', status),
        domain: ErrorDomain.MASTRA_VECTOR,
        category: ErrorCategory.USER,
        text,
        details: { indexName, ...details },
      });

    try {
      const inputs = { vectors, documents, sparseVectors, multiVectors };
      const lengths = Object.entries(inputs)
        .filter(([, value]) => value !== undefined)
        .map(([name, value]) => [name, (value as unknown[]).length] as const);
      if (lengths.length === 0 || lengths.every(([, length]) => length === 0)) {
        throw userError(
          'EMPTY_VECTORS',
          'Pass at least one document: vectors, documents, sparseVectors or multiVectors',
        );
      }
      const n = lengths[0]![1];
      for (const [name, length] of lengths) {
        if (length !== n) throw userError('LENGTH_MISMATCH', `${name} has ${length} entries, expected ${n}`);
      }
      if (metadata && metadata.length > 0 && metadata.length !== n) {
        throw userError('METADATA_LENGTH_MISMATCH', `metadata has ${metadata.length} entries, expected ${n}`);
      }
      if (ids && ids.length !== n) {
        throw userError('IDS_LENGTH_MISMATCH', `ids has ${ids.length} entries, expected ${n}`);
      }
      if (vectors) validateVectorValues(STORE, vectors);
      if (multiVectors) multiVectors.forEach(m => validateVectorValues(STORE, m));
      if (documents && documents.some(d => typeof d !== 'string')) {
        throw userError('INVALID_DOCUMENT', 'documents must be strings');
      }

      const info = await this.requireInfo(indexName, 'UPSERT');
      if (vectors && info.dimension === undefined) {
        throw userError(
          'NO_VECTOR_FIELD',
          `Index "${indexName}" was created without a dimension, so TopK embeds the text itself: pass documents instead of vectors`,
        );
      }
      if (vectors) {
        const bad = vectors.findIndex(v => v.length !== info.dimension);
        if (bad >= 0) {
          throw userError(
            'DIMENSION_MISMATCH',
            `Vector at index ${bad} has ${vectors[bad]!.length} dimensions, but index "${indexName}" has ${info.dimension}`,
          );
        }
      }
      if (sparseVectors && !info.sparse) {
        throw userError(
          'NO_SPARSE_FIELD',
          `Index "${indexName}" has no sparse vector field: create it with sparse: true`,
        );
      }
      if (multiVectors && info.multiVectorDimension === undefined) {
        throw userError(
          'NO_MULTI_VECTOR_FIELD',
          `Index "${indexName}" has no multi-vector field: create it with multiVector: { dimension }`,
        );
      }

      const outIds = ids ? ids.map(String) : Array.from({ length: n }, () => crypto.randomUUID());
      const rows: Row[] = [];
      const sizes: number[] = [];
      for (let i = 0; i < n; i++) {
        const meta = metadata?.[i] ?? {};
        const docText = documents?.[i] ?? (typeof meta?.text === 'string' ? meta.text : undefined);
        if (docText === undefined && !vectors && !sparseVectors && !multiVectors) {
          throw userError('MISSING_DOCUMENT', `Document at index ${i} has no text to search`);
        }
        if (info.textIndex === 'semantic' && docText !== undefined && docText.length > TOPK_MAX_SEMANTIC_TEXT_LENGTH) {
          throw userError(
            'TEXT_TOO_LONG',
            `Document at index ${i} has ${docText.length} characters, but a semantic index accepts at most ${TOPK_MAX_SEMANTIC_TEXT_LENGTH}. ` +
              'Split it into smaller chunks (for Memory, set semanticRecall.maxChunkChars: 4000), or use textIndex: "keyword".',
          );
        }
        const { row, size } = buildRow({
          id: outIds[i]!,
          metadata: meta,
          text: docText,
          vector: vectors?.[i],
          sparse: sparseVectors?.[i],
          multi: multiVectors?.[i],
        });
        rows.push(row);
        sizes.push(size);
      }

      if (deleteFilter && !this.translator.isEmptyFilter(deleteFilter)) {
        const expr = this.translator.translate(deleteFilter);
        if (expr) await this.deleteWhere(indexName, params.partition, expr);
      }
      await this.write(indexName, params.partition, rows, sizes);
      return outIds;
    } catch (error) {
      throw this.wrap(error, 'UPSERT', { indexName });
    }
  }

  /**
   * Updates the vector, metadata or text of one document by `id`, or of every document matching
   * `filter`. New metadata replaces the old metadata. With a new `document` in a `semantic` index,
   * TopK embeds the new text.
   */
  async updateVector(params: TopKUpdateVectorParams): Promise<void> {
    const { indexName, id, filter, update } = params;
    const userError = (status: string, text: string) =>
      new MastraError({
        id: createVectorErrorId(STORE, 'UPDATE_VECTOR', status),
        domain: ErrorDomain.MASTRA_VECTOR,
        category: ErrorCategory.USER,
        text,
        details: { indexName, ...(id ? { id } : {}) },
      });

    try {
      if (id && filter) throw userError('MUTUALLY_EXCLUSIVE', 'id and filter are mutually exclusive');
      if (!id && !filter) throw userError('NO_TARGET', 'Either id or filter must be provided');
      if (!update || (update.vector === undefined && update.metadata === undefined && update.document === undefined)) {
        throw userError('NO_PAYLOAD', 'No updates provided: pass vector, metadata or document');
      }
      if (filter && this.translator.isEmptyFilter(filter)) {
        throw userError('EMPTY_FILTER', 'Cannot use an empty filter: pass at least one condition');
      }
      if (update.vector) validateVectorValues(STORE, [update.vector]);

      const info = await this.requireInfo(indexName, 'UPDATE_VECTOR');
      if (update.vector && info.dimension === undefined) {
        throw userError('NO_VECTOR_FIELD', `Index "${indexName}" has no dense vector field`);
      }
      if (update.vector && update.vector.length !== info.dimension) {
        throw userError(
          'DIMENSION_MISMATCH',
          `Vector has ${update.vector.length} dimensions, but index "${indexName}" has ${info.dimension}`,
        );
      }
      if (
        update.document !== undefined &&
        info.textIndex === 'semantic' &&
        update.document.length > TOPK_MAX_SEMANTIC_TEXT_LENGTH
      ) {
        throw userError(
          'TEXT_TOO_LONG',
          `A semantic index accepts at most ${TOPK_MAX_SEMANTIC_TEXT_LENGTH} characters per document`,
        );
      }

      const ids = id ? [id] : await this.idsWhere(indexName, params, this.translator.translate(filter)!);
      if (ids.length === 0) {
        this.logger?.info(`No vectors matched the update in ${indexName}`);
        return;
      }

      // TopK's update() merges structs field by field, which can't remove metadata keys. To replace
      // metadata, read each document and write it back whole.
      const fields: string[] = [TOPK_FIELDS.text, TOPK_FIELDS.metadataJson, TOPK_FIELDS.metadata];
      if (info.dimension !== undefined) fields.push(TOPK_FIELDS.vector);
      if (info.sparse) fields.push(TOPK_FIELDS.sparseVector);
      if (info.multiVectorDimension !== undefined) fields.push(TOPK_FIELDS.multiVector);

      for (let start = 0; start < ids.length; start += 100) {
        const batch = ids.slice(start, start + 100);
        const found = await this.getDocuments(indexName, params, batch, fields);
        const rows: Row[] = [];
        const sizes: number[] = [];
        for (const docId of batch) {
          const existing = found[docId];
          if (!existing) continue;
          const oldMetadata = readMetadata(existing);
          const newMetadata = update.metadata ?? oldMetadata;
          let docText =
            typeof existing[TOPK_FIELDS.text] === 'string' ? (existing[TOPK_FIELDS.text] as string) : undefined;
          if (update.document !== undefined) {
            docText = update.document;
          } else if (update.metadata && docText !== undefined && docText === oldMetadata.text) {
            // The text was taken from metadata.text, so keep following it.
            docText = typeof update.metadata.text === 'string' ? update.metadata.text : docText;
          }
          const { row, size } = buildRow({
            id: docId,
            metadata: newMetadata,
            text: docText,
            vector: update.vector ?? (existing[TOPK_FIELDS.vector] as number[] | undefined),
            sparse: toSparse(existing[TOPK_FIELDS.sparseVector]),
            multi: existing[TOPK_FIELDS.multiVector] as number[][] | undefined,
          });
          rows.push(row);
          sizes.push(size);
        }
        if (rows.length > 0) await this.write(indexName, params.partition, rows, sizes);
      }
    } catch (error) {
      throw this.wrap(error, 'UPDATE_VECTOR', { indexName, ...(id ? { id } : {}) });
    }
  }

  async deleteVector({ indexName, id, partition }: TopKDeleteVectorParams): Promise<void> {
    try {
      await this.requireInfo(indexName, 'DELETE_VECTOR');
      await this.deleteWhere(indexName, partition, [id]);
    } catch (error) {
      throw this.wrap(error, 'DELETE_VECTOR', { indexName, id });
    }
  }

  async deleteVectors({ indexName, ids, filter, partition }: TopKDeleteVectorsParams): Promise<void> {
    const userError = (status: string, text: string) =>
      new MastraError({
        id: createVectorErrorId(STORE, 'DELETE_VECTORS', status),
        domain: ErrorDomain.MASTRA_VECTOR,
        category: ErrorCategory.USER,
        text,
        details: { indexName },
      });
    try {
      if (ids && filter) throw userError('MUTUALLY_EXCLUSIVE', 'ids and filter are mutually exclusive');
      if (!ids && !filter) throw userError('NO_TARGET', 'Either filter or ids must be provided');
      if (ids && ids.length === 0) throw userError('EMPTY_IDS', 'Cannot delete with an empty ids array');
      if (filter && this.translator.isEmptyFilter(filter)) {
        throw userError('EMPTY_FILTER', 'Cannot use an empty filter: pass at least one condition');
      }
      await this.requireInfo(indexName, 'DELETE_VECTORS');
      if (ids) {
        for (let start = 0; start < ids.length; start += 1000) {
          await this.deleteWhere(indexName, partition, ids.slice(start, start + 1000).map(String));
        }
      } else {
        await this.deleteWhere(indexName, partition, this.translator.translate(filter)!);
      }
    } catch (error) {
      throw this.wrap(error, 'DELETE_VECTORS', { indexName, ...(ids ? { idsCount: ids.length } : {}) });
    }
  }

  // ---------------------------------------------------------------------------
  // Query
  // ---------------------------------------------------------------------------

  /**
   * Searches an index. Pass `queryText` for semantic search embedded on the server, `queryVector`
   * for your own embeddings, both (or `mode: 'hybrid'`) to combine them with BM25 in one query, or
   * only a `filter` to list matching documents. See {@link TopKQueryVectorParams}.
   *
   * Scores are higher-is-better. Semantic and BM25 scores aren't normalized; for euclidean vector
   * search the score is `1 / (1 + distance)`.
   */
  async query(params: TopKQueryVectorParams): Promise<QueryResult[]> {
    const { indexName, topK = 10, filter, includeVector = false } = params;
    try {
      validateTopK(STORE, topK);
      if (topK > TOPK_MAX_RESULTS) {
        throw new MastraError({
          id: createVectorErrorId(STORE, 'QUERY', 'INVALID_TOP_K'),
          domain: ErrorDomain.MASTRA_VECTOR,
          category: ErrorCategory.USER,
          text: `topK must be at most ${TOPK_MAX_RESULTS}`,
          details: { indexName, topK },
        });
      }
      if (params.queryVector) validateVectorValues(STORE, [params.queryVector]);
      if (params.multiVector) validateVectorValues(STORE, params.multiVector);

      const info = await this.requireInfo(indexName, 'QUERY');
      const plan = this.planQuery(indexName, info, params);
      const metadataFilter = this.translator.translate(filter);

      let q = select({
        ...(plan.score ? { score: plan.score } : {}),
        [TOPK_FIELDS.text]: field(TOPK_FIELDS.text),
        [TOPK_FIELDS.metadataJson]: field(TOPK_FIELDS.metadataJson),
      });
      // A text expression and a filter expression must be separate filter stages.
      if (plan.textFilter) q = q.filter(plan.textFilter);
      if (metadataFilter) q = q.filter(metadataFilter);
      if (plan.score) q = q.sort(field('score'), plan.ascending);
      q = q.limit(topK);

      const options = this.readOptions(indexName, params);
      let rows: Row[];
      try {
        rows = await this.collection(indexName, params.partition).query(q, options);
      } catch (error) {
        if (hasMessage(error, 'partition not found')) return [];
        throw error;
      }
      if (rows.length === 0) return [];

      const ids = rows.map(r => String(r._id));
      const extraFields: string[] = [];
      if (includeVector && info.dimension !== undefined) extraFields.push(TOPK_FIELDS.vector);
      if (rows.some(r => typeof r[TOPK_FIELDS.metadataJson] !== 'string')) extraFields.push(TOPK_FIELDS.metadata);
      // A query can't return an indexed vector field or a whole struct, so read them by ID.
      const extra = extraFields.length > 0 ? await this.getDocuments(indexName, params, ids, extraFields) : {};

      return rows.map(row => {
        const docId = String(row._id);
        const result: QueryResult = {
          id: docId,
          score: plan.toScore(typeof row.score === 'number' ? row.score : 0),
          metadata: readMetadata({ ...extra[docId], ...row }),
        };
        if (typeof row[TOPK_FIELDS.text] === 'string') result.document = row[TOPK_FIELDS.text] as string;
        const vector = extra[docId]?.[TOPK_FIELDS.vector];
        if (includeVector && Array.isArray(vector)) result.vector = vector as number[];
        return result;
      });
    } catch (error) {
      throw this.wrap(error, 'QUERY', { indexName });
    }
  }

  private planQuery(
    indexName: string,
    info: CollectionInfo,
    params: TopKQueryVectorParams,
  ): {
    score?: query.FunctionExpression | query.LogicalExpression;
    textFilter?: query.TextExpression;
    ascending: boolean;
    toScore: (raw: number) => number;
  } {
    const { queryVector, queryText, sparseVector, multiVector, mode, weights, scoring, keywordMatchAll } = params;
    const keywordText = params.keywordQuery ?? queryText;
    const userError = (status: string, text: string) =>
      new MastraError({
        id: createVectorErrorId(STORE, 'QUERY', status),
        domain: ErrorDomain.MASTRA_VECTOR,
        category: ErrorCategory.USER,
        text,
        details: { indexName, ...(mode ? { mode } : {}) },
      });

    if (queryVector && info.dimension === undefined) {
      throw userError('NO_VECTOR_FIELD', `Index "${indexName}" has no dense vector field: query it with queryText`);
    }
    if (queryVector && queryVector.length !== info.dimension) {
      throw userError(
        'DIMENSION_MISMATCH',
        `Query vector has ${queryVector.length} dimensions, but index "${indexName}" has ${info.dimension}`,
      );
    }

    // Signals this query has input for, and this collection can score.
    const available: TopKSignal[] = [];
    if (queryText && info.textIndex === 'semantic') available.push('semantic');
    if (keywordText && info.textIndex !== 'none') available.push('keyword');
    if (queryVector) available.push('vector');
    if (sparseVector && info.sparse) available.push('sparse');
    if (multiVector && info.multiVectorDimension !== undefined) available.push('multiVector');

    const requireSignal = (signal: TopKSignal) => {
      if (available.includes(signal)) return;
      const why: Record<TopKSignal, string> = {
        semantic: info.textIndex === 'semantic' ? 'pass queryText' : `index "${indexName}" has no semantic text index`,
        keyword:
          info.textIndex === 'none' ? `index "${indexName}" has no text index` : 'pass queryText or keywordQuery',
        vector: 'pass queryVector',
        sparse: info.sparse ? 'pass sparseVector' : `index "${indexName}" has no sparse vector field`,
        multiVector:
          info.multiVectorDimension !== undefined
            ? 'pass multiVector'
            : `index "${indexName}" has no multi-vector field`,
      };
      throw userError('MISSING_SIGNAL', `${signal} search needs input: ${why[signal]}`);
    };

    let resolved: TopKSearchMode | undefined = mode;
    if (!resolved && !scoring) {
      if (weights || available.length > 1) resolved = 'hybrid';
      else resolved = available[0];
    }
    if (mode && !['semantic', 'keyword', 'vector', 'sparse', 'multiVector', 'hybrid'].includes(mode)) {
      throw userError('INVALID_MODE', `Unknown mode "${mode}"`);
    }

    let signals: TopKSignal[];
    if (scoring) {
      if (mode) throw userError('MODE_WITH_SCORING', 'Pass either mode or scoring, not both');
      signals = available;
      if (signals.length === 0) throw userError('MISSING_SIGNAL', 'scoring needs at least one query input');
    } else if (!resolved) {
      // Filter-only query: list matching documents.
      return { ascending: false, toScore: () => 0 };
    } else if (resolved === 'hybrid') {
      if (weights) {
        for (const [key, weight] of Object.entries(weights)) {
          if (!['semantic', 'keyword', 'vector', 'sparse', 'multiVector'].includes(key)) {
            throw userError('INVALID_WEIGHTS', `Unknown hybrid weight "${key}"`);
          }
          if (typeof weight !== 'number' || !Number.isFinite(weight)) {
            throw userError('INVALID_WEIGHTS', `Weight "${key}" must be a finite number`);
          }
        }
        signals = (Object.keys(weights) as TopKSignal[]).filter(key => weights[key] !== 0);
        signals.forEach(requireSignal);
      } else {
        signals = available;
      }
      if (signals.length === 0) throw userError('MISSING_SIGNAL', 'hybrid search needs at least one query input');
    } else {
      requireSignal(resolved);
      signals = [resolved];
    }

    const euclidean = info.metric === 'euclidean';
    const expressions: Partial<Record<TopKSignal, query.FunctionExpression>> = {};
    let textFilter: query.TextExpression | undefined;
    for (const signal of signals) {
      switch (signal) {
        case 'semantic':
          expressions.semantic = fn.semanticSimilarity(TOPK_FIELDS.text, queryText!);
          break;
        case 'keyword':
          expressions.keyword = fn.bm25Score();
          textFilter =
            resolved === 'keyword'
              ? match(keywordText!, { field: TOPK_FIELDS.text, all: keywordMatchAll ?? false })
              : // Boost documents that contain query terms without dropping the ones that don't.
                should(keywordText!, { field: TOPK_FIELDS.text });
          break;
        case 'vector':
          expressions.vector = fn.vectorDistance(TOPK_FIELDS.vector, queryVector!);
          break;
        case 'sparse':
          expressions.sparse = fn.vectorDistance(
            TOPK_FIELDS.sparseVector,
            f32SparseVector(sparseRecord(sparseVector!)),
          );
          break;
        case 'multiVector':
          expressions.multiVector = fn.multiVectorDistance(TOPK_FIELDS.multiVector, matrix(multiVector!, 'f32'));
          break;
      }
    }

    if (scoring) {
      // A function expression can't take another one as an operand, so hand out plain expressions.
      const signalsForScoring: TopKScoreSignals = {};
      for (const [signal, expr] of Object.entries(expressions) as [TopKSignal, query.FunctionExpression][]) {
        signalsForScoring[signal] = signal === 'keyword' ? expr.coalesce(0) : expr.mul(1);
      }
      const score = scoring(signalsForScoring);
      if (!score || typeof (score as { add?: unknown }).add !== 'function') {
        throw userError('INVALID_SCORING', 'scoring must return a TopK expression');
      }
      return { score, textFilter, ascending: false, toScore: raw => raw };
    }

    if (signals.length === 1 && resolved !== 'hybrid') {
      const only = signals[0]!;
      if (only === 'vector' && euclidean) {
        // TopK returns the squared euclidean distance: lower is better.
        return {
          score: expressions.vector,
          textFilter,
          ascending: true,
          toScore: raw => 1 / (1 + Math.sqrt(Math.max(raw, 0))),
        };
      }
      return { score: expressions[only], textFilter, ascending: false, toScore: raw => raw };
    }

    // Weighted sum, computed in one select. A distance (euclidean) counts negatively.
    let combined: query.FunctionExpression | query.LogicalExpression | undefined;
    for (const signal of signals) {
      const weight = (weights?.[signal] ?? 1) * (signal === 'vector' && euclidean ? -1 : 1);
      const expr = expressions[signal]!;
      const term = signal === 'keyword' ? expr.coalesce(0).mul(weight) : expr.mul(weight);
      combined = combined ? combined.add(term) : term;
    }
    return { score: combined, textFilter, ascending: false, toScore: raw => raw };
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private collectionName(indexName: string): string {
    return `${this.config.indexPrefix ?? ''}${indexName}`;
  }

  private collection(indexName: string, partition?: string) {
    const p = partition ?? this.partition;
    return this.client.collection(this.collectionName(indexName), p ?? undefined);
  }

  private lsnKey(indexName: string, partition?: string): string {
    return `${indexName}\u0000${partition ?? ''}`;
  }

  private readOptions(indexName: string, params: TopKReadParams): QueryOptions | undefined {
    if (params.lsn) return { lsn: params.lsn };
    if (params.consistency) return { consistency: params.consistency };
    if (this.config.readYourWrites !== false) {
      const lsn = this.shared.lsns.get(this.lsnKey(indexName, params.partition ?? this.partition));
      if (lsn) return { lsn };
    }
    if (this.config.consistency) return { consistency: this.config.consistency };
    return undefined;
  }

  private recordLsn(indexName: string, partition: string | undefined, lsn: string) {
    if (lsn) this.shared.lsns.set(this.lsnKey(indexName, partition ?? this.partition), lsn);
  }

  private forget(indexName: string) {
    this.shared.infos.delete(indexName);
    for (const key of Array.from(this.shared.lsns.keys())) {
      if (key.startsWith(`${indexName}\u0000`)) this.shared.lsns.delete(key);
    }
  }

  private async write(indexName: string, partition: string | undefined, rows: Row[], sizes: number[]) {
    let batch: Row[] = [];
    let total = 0;
    const flush = async () => {
      if (batch.length === 0) return;
      this.recordLsn(indexName, partition, await this.collection(indexName, partition).upsert(batch));
      batch = [];
      total = 0;
    };
    for (let i = 0; i < rows.length; i++) {
      if (batch.length > 0 && (total + sizes[i]! > MAX_REQUEST_BYTES || batch.length >= 500)) await flush();
      batch.push(rows[i]!);
      total += sizes[i]!;
    }
    await flush();
  }

  private async deleteWhere(
    indexName: string,
    partition: string | undefined,
    target: string[] | query.LogicalExpression,
  ) {
    try {
      this.recordLsn(indexName, partition, await this.collection(indexName, partition).delete(target));
    } catch (error) {
      // Nothing to delete in a partition that was never written.
      if (!hasMessage(error, 'partition not found')) throw error;
    }
  }

  private async getDocuments(
    indexName: string,
    params: TopKReadParams,
    ids: string[],
    fields: string[],
  ): Promise<Record<string, Row>> {
    try {
      return await this.collection(indexName, params.partition).get(ids, fields, this.readOptions(indexName, params));
    } catch (error) {
      if (hasMessage(error, 'partition not found')) return {};
      throw error;
    }
  }

  /** IDs of every document matching a filter, read in pages of 10,000 (TopK's query limit). */
  private async idsWhere(indexName: string, params: TopKReadParams, expr: query.LogicalExpression): Promise<string[]> {
    const ids: string[] = [];
    const options = this.readOptions(indexName, params);
    for (let offset = 0; ; offset += TOPK_MAX_RESULTS) {
      let rows: Row[];
      try {
        rows = await this.collection(indexName, params.partition).query(
          select({}).filter(expr).sort(field('_id'), true).offset(offset).limit(TOPK_MAX_RESULTS),
          options,
        );
      } catch (error) {
        if (hasMessage(error, 'partition not found')) return ids;
        throw error;
      }
      ids.push(...rows.map(r => String(r._id)));
      if (rows.length < TOPK_MAX_RESULTS) return ids;
    }
  }

  /** Reads a collection's layout from its schema, or `undefined` if it doesn't exist. Cached. */
  private async loadInfo(indexName: string): Promise<CollectionInfo | undefined> {
    const cached = this.shared.infos.get(indexName);
    if (cached) return cached;
    let schema: Record<string, CollectionFieldSpec>;
    try {
      schema = (await this.client.collections().get(this.collectionName(indexName))).schema;
    } catch (error) {
      if (hasMessage(error, 'collection not found')) return undefined;
      throw error;
    }
    const info = parseSchema(schema);
    this.shared.infos.set(indexName, info);
    return info;
  }

  private async requireInfo(indexName: string, operation: string): Promise<CollectionInfo> {
    this.assertIndexName(indexName, operation);
    const info = await this.loadInfo(indexName);
    if (!info) {
      throw new MastraError({
        id: createVectorErrorId(STORE, operation, 'INDEX_NOT_FOUND'),
        domain: ErrorDomain.MASTRA_VECTOR,
        category: ErrorCategory.USER,
        text: `Index "${indexName}" not found. Create it with createIndex() first.`,
        details: { indexName },
      });
    }
    return info;
  }

  private checkCompatible(
    indexName: string,
    existing: CollectionInfo,
    requested: CollectionInfo,
    params: TopKCreateIndexParams,
  ) {
    const mismatch = (status: string, text: string) =>
      new MastraError({
        id: createVectorErrorId(STORE, 'CREATE_INDEX', status),
        domain: ErrorDomain.MASTRA_VECTOR,
        category: ErrorCategory.USER,
        text: `Index "${indexName}" already exists ${text}. Use a new index name, or delete this one first.`,
        details: { indexName },
      });
    if (params.dimension !== undefined && existing.dimension !== params.dimension) {
      throw mismatch(
        'DIMENSION_MISMATCH',
        existing.dimension === undefined
          ? `without a dense vector field (TopK embeds its text), but a dimension of ${params.dimension} was requested`
          : `with ${existing.dimension} dimensions, but ${params.dimension} were requested`,
      );
    }
    if (params.textIndex !== undefined && existing.textIndex !== params.textIndex) {
      throw mismatch(
        'TEXT_INDEX_MISMATCH',
        `with text index "${existing.textIndex}", but "${params.textIndex}" was requested`,
      );
    }
    if (params.dimension === undefined && params.textIndex === undefined && existing.textIndex !== 'semantic') {
      // createIndex({ indexName }) asks for a collection TopK embeds itself.
      if (existing.dimension === undefined) {
        throw mismatch('TEXT_INDEX_MISMATCH', `with text index "${existing.textIndex}", which TopK can't embed`);
      }
    }
    if (params.sparse && !existing.sparse) throw mismatch('SPARSE_MISMATCH', 'without a sparse vector field');
    if (params.multiVector && existing.multiVectorDimension !== params.multiVector.dimension) {
      throw mismatch('MULTI_VECTOR_MISMATCH', 'with a different multi-vector field');
    }
    if (requested.metric && existing.metric && params.metric && existing.metric !== params.metric) {
      this.logger?.warn(
        `Index "${indexName}" already exists with metric "${existing.metric}", so metric "${params.metric}" is ignored. To change it, use a new index.`,
      );
    }
  }

  private assertIndexName(indexName: string, operation: string) {
    if (!indexName || typeof indexName !== 'string') {
      throw new MastraError({
        id: createVectorErrorId(STORE, operation, 'INVALID_INDEX_NAME'),
        domain: ErrorDomain.MASTRA_VECTOR,
        category: ErrorCategory.USER,
        text: 'indexName must be a non-empty string',
      });
    }
  }

  private wrap(error: unknown, operation: string, details: Record<string, any> = {}): MastraError {
    if (error instanceof MastraError) return error;
    const message = error instanceof Error ? error.message : String(error);
    const user =
      /DocumentValidationError|SchemaValidationError|invalid argument|Unsupported|requires|must|Filter|filter|operator/i.test(
        message,
      ) && !/permission denied|unauthenticated/i.test(message);
    return new MastraError(
      {
        id: createVectorErrorId(STORE, operation, 'FAILED'),
        domain: ErrorDomain.MASTRA_VECTOR,
        category: user ? ErrorCategory.USER : ErrorCategory.THIRD_PARTY,
        text: message,
        details,
      },
      error,
    );
  }
}

// -----------------------------------------------------------------------------
// Document encoding
// -----------------------------------------------------------------------------

/**
 * Builds a TopK document. Metadata is written three ways: as a struct for filters, as JSON for
 * results (a query can't return a whole struct), and as the list of its key paths and array
 * lengths, so `$exists`, `null` and array filters are exact.
 */
function buildRow(input: {
  id: string;
  metadata: Record<string, any> | null | undefined;
  text?: string;
  vector?: number[];
  sparse?: SparseVector;
  multi?: number[][];
}): { row: Row; size: number } {
  const json = toJson(input.metadata ?? {});
  const parsed = JSON.parse(json) as Record<string, unknown>;
  const keys: string[] = [];
  const sizes: Record<string, unknown> = {};
  const struct = encodeStruct(parsed, '', keys, sizes);

  const row: Row = {
    _id: input.id,
    [TOPK_FIELDS.metadata]: struct,
    [TOPK_FIELDS.metadataJson]: json,
    [TOPK_FIELDS.metadataKeys]: stringList(keys),
    [TOPK_FIELDS.metadataSizes]: sizes,
  };
  let size = 256 + input.id.length + json.length * 2 + keys.join('').length;
  if (input.text !== undefined) {
    row[TOPK_FIELDS.text] = input.text;
    size += Buffer.byteLength(input.text);
  }
  if (input.vector) {
    row[TOPK_FIELDS.vector] = input.vector;
    size += input.vector.length * 12;
  }
  if (input.sparse) {
    row[TOPK_FIELDS.sparseVector] = f32SparseVector(sparseRecord(input.sparse));
    size += input.sparse.indices.length * 16;
  }
  if (input.multi) {
    row[TOPK_FIELDS.multiVector] = matrix(input.multi, 'f32');
    size += input.multi.reduce((sum, r) => sum + r.length * 12, 0);
  }
  return { row, size };
}

function toJson(metadata: Record<string, any>): string {
  if (typeof metadata !== 'object' || Array.isArray(metadata)) {
    throw new Error('metadata must be an object');
  }
  let json: string | undefined;
  try {
    json = JSON.stringify(metadata);
  } catch (error) {
    throw new Error(`metadata must be JSON-serializable: ${(error as Error).message}`);
  }
  return json ?? '{}';
}

function encodeStruct(
  value: Record<string, unknown>,
  prefix: string,
  keys: string[],
  sizes: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, v] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${key}` : key;
    keys.push(path);
    if (Array.isArray(v)) {
      sizes[key] = v.length;
      if (v.every(x => typeof x === 'string')) out[key] = stringList(v as string[]);
      else if (v.every(x => typeof x === 'number')) {
        const numbers = v as number[];
        out[key] = numbers.every(Number.isInteger) ? i64List(numbers) : f64List(numbers);
      }
      // Arrays of booleans, objects or mixed types can't be filtered in TopK. They stay in the JSON copy.
    } else if (v !== null && typeof v === 'object') {
      const nestedSizes: Record<string, unknown> = {};
      out[key] = encodeStruct(v as Record<string, unknown>, path, keys, nestedSizes);
      if (Object.keys(nestedSizes).length > 0) sizes[key] = nestedSizes;
    } else {
      out[key] = v;
    }
  }
  return out;
}

function readMetadata(row: Row): Record<string, any> {
  const json = row[TOPK_FIELDS.metadataJson];
  if (typeof json === 'string') {
    try {
      return JSON.parse(json);
    } catch {
      // Fall back to the struct.
    }
  }
  const struct = row[TOPK_FIELDS.metadata];
  return struct && typeof struct === 'object' ? (struct as Record<string, any>) : {};
}

function sparseRecord(vector: SparseVector): Record<number, number> {
  if (!vector || vector.indices.length !== vector.values.length) {
    throw new Error('A sparse vector needs as many indices as values');
  }
  const record: Record<number, number> = {};
  vector.indices.forEach((index, i) => {
    const value = vector.values[i]!;
    if (!Number.isInteger(index) || index < 0 || !Number.isFinite(value)) {
      throw new Error('Sparse vector indices must be non-negative integers and values finite numbers');
    }
    record[index] = value;
  });
  return record;
}

function toSparse(value: unknown): SparseVector | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const entries = Object.entries(value as Record<string, number>);
  return { indices: entries.map(([k]) => Number(k)), values: entries.map(([, v]) => v) };
}

function parseSchema(schema: Record<string, CollectionFieldSpec>): CollectionInfo {
  const textSpec = schema[TOPK_FIELDS.text];
  const textIndexType = textSpec?.index?.type;
  const textIndex: TopKTextIndex =
    textIndexType === 'SemanticIndex' ? 'semantic' : textIndexType === 'KeywordIndex' ? 'keyword' : 'none';
  const info: CollectionInfo = { textIndex, sparse: !!schema[TOPK_FIELDS.sparseVector] };
  const vectorSpec = schema[TOPK_FIELDS.vector];
  if (vectorSpec?.dataType && 'dimension' in vectorSpec.dataType) {
    info.dimension = vectorSpec.dataType.dimension;
    const metric = vectorSpec.index?.type === 'VectorIndex' ? vectorSpec.index.metric : undefined;
    info.metric = metric === 'dot_product' ? 'dotproduct' : metric === 'euclidean' ? 'euclidean' : 'cosine';
  }
  const multiSpec = schema[TOPK_FIELDS.multiVector];
  if (multiSpec?.dataType && 'dimension' in multiSpec.dataType) {
    info.multiVectorDimension = multiSpec.dataType.dimension;
  }
  return info;
}

function hasMessage(error: unknown, text: string): boolean {
  let current: unknown = error;
  const seen = new Set<unknown>();
  while (current && !seen.has(current)) {
    seen.add(current);
    if (current instanceof Error && current.message.toLowerCase().includes(text)) return true;
    current = (current as { cause?: unknown }).cause;
  }
  return false;
}
