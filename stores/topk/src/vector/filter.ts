import type {
  BlacklistedRootOperators,
  LogicalOperatorValueMap,
  OperatorSupport,
  OperatorValueMap,
  VectorFilter,
} from '@mastra/core/vector/filter';
import { BaseFilterTranslator } from '@mastra/core/vector/filter';
import type { query } from 'topk-js';
import { f64List } from 'topk-js/data';
import { all, any, field, literal, not, LogicalExpression } from 'topk-js/query';

/** Internal document fields written by `TopKVector`. */
export const TOPK_FIELDS = {
  /** Document text: semantic and/or keyword (BM25) indexed. */
  text: 'text',
  /** Dense vector. */
  vector: 'vector',
  /** Sparse vector. */
  sparseVector: 'sparse_vector',
  /** Multi-vector (late interaction) matrix. */
  multiVector: 'multi_vector',
  /** Metadata as a struct, used for filtering (`metadata.<key>`). */
  metadata: 'metadata',
  /** Metadata as JSON text. A query can't return a whole struct, so results are read from this copy. */
  metadataJson: 'metadata_json',
  /** Every metadata key path, including nested ones, so `$exists` and `null` checks are exact. */
  metadataKeys: 'metadata_keys',
  /** Length of every array in the metadata, at the same path, for `$size` and array-aware `$in`. */
  metadataSizes: 'metadata_sizes',
} as const;

type TopKOperatorValueMap = Omit<OperatorValueMap, '$elemMatch'> & {
  $contains: string | number;
  $size: number;
};

export type TopKVectorFilter =
  | VectorFilter<
      keyof TopKOperatorValueMap,
      TopKOperatorValueMap,
      LogicalOperatorValueMap,
      BlacklistedRootOperators | '$contains' | '$size'
    >
  /** A TopK filter expression built with `topk-js/query`, passed through unchanged. */
  | query.LogicalExpression;

type Expr = query.LogicalExpression;
type Scalar = string | number | boolean | null;

const TRUE = () => literal(true);
const FALSE = () => literal(false);

/**
 * Builds a reference to a metadata field, for use in custom scoring expressions and raw TopK filters.
 *
 * @example
 * ```ts
 * import { field } from 'topk-js/query';
 * metadataField('year').gte(2020) // same as field('metadata.year').gte(2020)
 * ```
 */
export function metadataField(path: string): query.LogicalExpression {
  return field(`${TOPK_FIELDS.metadata}.${path}`);
}

/**
 * Translates Mastra's MongoDB-style metadata filters into TopK filter expressions.
 *
 * Metadata is stored in the `metadata` struct field, so `{ year: { $gte: 2020 } }` becomes
 * `field('metadata.year').gte(2020)`. Semantics match Mastra's other stores:
 *
 * - `$ne`, `$nin` and field-level `$not` match only documents where the field has a value,
 *   as in SQL stores such as `@mastra/pg`. Use `$exists: false` to find documents without it.
 * - `{ tags: 'a' }`, `$in` and `$nin` match an array field when any element matches.
 * - `$exists` and `{ field: null }` tell a missing key apart from a key set to `null`.
 *
 * `$elemMatch` isn't supported: TopK can't filter on the elements of an array of objects.
 */
export class TopKFilterTranslator extends BaseFilterTranslator<TopKVectorFilter, Expr | undefined> {
  protected override getSupportedOperators(): OperatorSupport {
    return {
      ...BaseFilterTranslator.DEFAULT_OPERATORS,
      array: ['$in', '$nin', '$all'],
      custom: ['$contains', '$size'],
    };
  }

  /** Whether a filter selects every document: `undefined`, `null` or `{}`. */
  isEmptyFilter(filter: TopKVectorFilter | undefined): boolean {
    return !(filter instanceof LogicalExpression) && this.isEmpty(filter);
  }

  translate(filter?: TopKVectorFilter): Expr | undefined {
    if (filter instanceof LogicalExpression) {
      return filter;
    }
    if (this.isEmpty(filter)) {
      return undefined;
    }
    if (typeof filter !== 'object' || Array.isArray(filter)) {
      throw new Error('Filter must be an object, for example { category: "news" }');
    }
    this.validateFilter(filter);
    return this.translateNode(filter as Record<string, unknown>);
  }

  private translateNode(node: Record<string, unknown>): Expr {
    const clauses: Expr[] = [];
    for (const [key, value] of Object.entries(node)) {
      if (this.isLogicalOperator(key)) {
        clauses.push(this.translateLogical(key, value));
      } else if (key.startsWith('$')) {
        throw new Error(`Unsupported operator: ${key}`);
      } else {
        clauses.push(this.translateField(key, value));
      }
    }
    return combineAll(clauses);
  }

  private translateLogical(operator: string, value: unknown): Expr {
    if (operator === '$not') {
      if (!value || typeof value !== 'object' || Array.isArray(value)) {
        throw new Error('$not operator requires an object');
      }
      return not(this.translateNode(value as Record<string, unknown>));
    }
    if (!Array.isArray(value)) {
      throw new Error(`Logical operator ${operator} requires an array of conditions`);
    }
    const clauses = value.map(condition => {
      if (!condition || typeof condition !== 'object' || Array.isArray(condition)) {
        throw new Error(`Invalid condition for logical operator ${operator}`);
      }
      return this.translateNode(condition as Record<string, unknown>);
    });
    switch (operator) {
      case '$and':
        return clauses.length === 0 ? TRUE() : combineAll(clauses);
      case '$or':
        return clauses.length === 0 ? FALSE() : combineAny(clauses);
      case '$nor':
        return clauses.length === 0 ? TRUE() : not(combineAny(clauses));
      default:
        throw new Error(`Unsupported logical operator: ${operator}`);
    }
  }

  private translateField(path: string, value: unknown): Expr {
    if (path === '') {
      throw new Error('Filter keys must be non-empty strings');
    }
    if (value instanceof RegExp) {
      return this.regex(path, value.source, value.flags);
    }
    if (value instanceof Date || this.isPrimitive(value)) {
      return this.eq(path, this.scalar(value));
    }
    if (Array.isArray(value)) {
      return this.in(path, value);
    }
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) {
      throw new Error(`Empty condition for filter key "${path}"`);
    }
    const operatorKeys = entries.filter(([key]) => key.startsWith('$'));
    if (operatorKeys.length === 0) {
      // A nested object without operators: match each nested key, `{ a: { b: 1 } }` -> `a.b == 1`.
      return combineAll(entries.map(([key, nested]) => this.translateField(`${path}.${key}`, nested)));
    }
    if (operatorKeys.length !== entries.length) {
      throw new Error(`Filter key "${path}" mixes operators and nested fields`);
    }
    const condition = value as Record<string, unknown>;
    const clauses: Expr[] = [];
    for (const [operator, operand] of entries) {
      if (operator === '$options') {
        if (!('$regex' in condition)) {
          throw new Error('$options requires $regex');
        }
        continue;
      }
      clauses.push(this.translateOperator(path, operator, operand, condition));
    }
    return combineAll(clauses);
  }

  private translateOperator(
    path: string,
    operator: string,
    operand: unknown,
    condition: Record<string, unknown>,
  ): Expr {
    const f = metadataField(path);
    switch (operator) {
      case '$eq':
        return this.eq(path, this.scalar(operand));
      case '$ne': {
        const value = this.scalar(operand);
        return value === null ? f.isNotNull() : f.isNotNull().and(not(this.eq(path, value)));
      }
      case '$gt':
        return f.gt(this.comparable(operator, operand));
      case '$gte':
        return f.gte(this.comparable(operator, operand));
      case '$lt':
        return f.lt(this.comparable(operator, operand));
      case '$lte':
        return f.lte(this.comparable(operator, operand));
      case '$in':
        return this.in(path, this.list(operator, operand));
      case '$nin':
        return f.isNotNull().and(not(this.in(path, this.list(operator, operand))));
      case '$all': {
        const values = this.list(operator, operand);
        if (values.length === 0) {
          return FALSE();
        }
        return isArray(path).and(combineAll(values.map(v => f.contains(this.containable(operator, v)))));
      }
      case '$exists':
        if (typeof operand !== 'boolean') {
          throw new Error('$exists requires a boolean');
        }
        return operand ? hasKey(path) : not(hasKey(path));
      case '$regex': {
        const options = condition.$options;
        if (options !== undefined && typeof options !== 'string') {
          throw new Error('$options must be a string');
        }
        if (operand instanceof RegExp) {
          return this.regex(path, operand.source, mergeFlags(operand.flags, options));
        }
        if (typeof operand !== 'string') {
          throw new Error('$regex requires a string or a RegExp');
        }
        return this.regex(path, operand, options);
      }
      case '$contains':
        return f.contains(this.containable(operator, operand));
      case '$size':
        if (typeof operand !== 'number' || !Number.isInteger(operand) || operand < 0) {
          throw new Error('$size requires a non-negative integer');
        }
        return field(`${TOPK_FIELDS.metadataSizes}.${path}`).eq(operand);
      case '$not': {
        if (operand instanceof RegExp) {
          return f.isNotNull().and(not(this.regex(path, operand.source, operand.flags)));
        }
        if (!operand || typeof operand !== 'object' || Array.isArray(operand) || Object.keys(operand).length === 0) {
          throw new Error('$not operator requires a non-empty object');
        }
        const negated = not(this.translateField(path, operand));
        // `$not: { $exists: ... }` is about the key itself, so it isn't limited to documents that have one.
        return '$exists' in operand ? negated : f.isNotNull().and(negated);
      }
      default:
        throw new Error(`Unsupported operator: ${operator}`);
    }
  }

  /** Equality, with MongoDB semantics for `null` (key present and null) and arrays (any element equals). */
  private eq(path: string, value: Scalar): Expr {
    const f = metadataField(path);
    if (value === null) {
      return hasKey(path).and(f.isNull());
    }
    if (typeof value === 'boolean') {
      // Boolean arrays aren't stored in the struct (TopK lists have no boolean type), so a boolean
      // only ever matches a scalar.
      return f.eq(value);
    }
    return f.eq(value).or(isArray(path).and(f.contains(value)));
  }

  /** `$in`: the value equals one of the values, or the array holds at least one of them. */
  private in(path: string, raw: unknown[]): Expr {
    const values = raw.map(v => this.scalar(v));
    if (values.length === 0) {
      return FALSE();
    }
    const f = metadataField(path);
    const clauses: Expr[] = [];
    const strings = values.filter((v): v is string => typeof v === 'string');
    const numbers = values.filter((v): v is number => typeof v === 'number');
    if (strings.length > 0) {
      clauses.push(f.in(strings));
    }
    if (numbers.length > 0) {
      // A list of numbers must be sent as f64: a plain array of mixed integers and floats, or of
      // floats alone, silently matches nothing.
      clauses.push(f.in(f64List(numbers)));
    }
    for (const v of values) {
      if (typeof v === 'boolean') {
        clauses.push(f.eq(v));
      } else if (v === null) {
        clauses.push(hasKey(path).and(f.isNull()));
      }
    }
    const members = values.filter((v): v is string | number => typeof v === 'string' || typeof v === 'number');
    if (members.length > 0) {
      clauses.push(isArray(path).and(combineAny(members.map(v => f.contains(v)))));
    }
    return combineAny(clauses);
  }

  private regex(path: string, pattern: string, flags?: string): Expr {
    return metadataField(path).regexpMatch(pattern, flags || undefined);
  }

  private scalar(value: unknown): Scalar {
    if (value === undefined) {
      return null;
    }
    if (value instanceof Date) {
      return value.toISOString();
    }
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) {
        throw new Error(`Filter values must be finite numbers, got ${value}`);
      }
      return Object.is(value, -0) ? 0 : value;
    }
    if (value === null || typeof value === 'string' || typeof value === 'boolean') {
      return value;
    }
    throw new Error(`Unsupported filter value: ${JSON.stringify(value)}`);
  }

  private comparable(operator: string, value: unknown): number | string {
    const normalized = this.scalar(value);
    if (typeof normalized !== 'number' && typeof normalized !== 'string') {
      throw new Error(`${operator} requires a number, string or Date`);
    }
    return normalized;
  }

  private containable(operator: string, value: unknown): string | number {
    const normalized = this.scalar(value);
    if (typeof normalized !== 'string' && typeof normalized !== 'number') {
      throw new Error(`${operator} requires a string or a number`);
    }
    return normalized;
  }

  private list(operator: string, value: unknown): unknown[] {
    if (!Array.isArray(value)) {
      throw new Error(`${operator} requires an array`);
    }
    return value;
  }
}

function hasKey(path: string): Expr {
  return field(TOPK_FIELDS.metadataKeys).contains(path);
}

function isArray(path: string): Expr {
  return field(`${TOPK_FIELDS.metadataSizes}.${path}`).isNotNull();
}

function combineAll(clauses: Expr[]): Expr {
  if (clauses.length === 1) return clauses[0]!;
  return all(clauses);
}

function combineAny(clauses: Expr[]): Expr {
  if (clauses.length === 1) return clauses[0]!;
  return any(clauses);
}

function mergeFlags(a: string, b?: string): string {
  return Array.from(new Set(`${a}${b ?? ''}`)).join('');
}
