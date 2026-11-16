import { f64List } from 'topk-js/data';
import { all, any, field, literal, not } from 'topk-js/query';
import { describe, expect, it } from 'vitest';
import { TopKFilterTranslator, metadataField } from './filter';
import type { TopKVectorFilter } from './filter';

const translator = new TopKFilterTranslator();
const t = (filter: TopKVectorFilter) => String(translator.translate(filter));
const m = metadataField;
const keys = () => field('metadata_keys');
const isArray = (path: string) => field(`metadata_sizes.${path}`).isNotNull();
const eq = (path: string, value: string | number) =>
  m(path)
    .eq(value)
    .or(isArray(path).and(m(path).contains(value)));

describe('TopKFilterTranslator', () => {
  describe('empty filters', () => {
    it('returns undefined for no filter', () => {
      expect(translator.translate(undefined)).toBeUndefined();
      expect(translator.translate(null)).toBeUndefined();
      expect(translator.translate({})).toBeUndefined();
    });

    it('passes a TopK expression through unchanged', () => {
      const expr = field('metadata.year').gte(2020);
      expect(translator.translate(expr)).toBe(expr);
    });
  });

  describe('equality', () => {
    it('translates a direct value, matching a scalar or an array element', () => {
      expect(t({ category: 'news' })).toBe(String(eq('category', 'news')));
      expect(t({ count: { $eq: 5 } })).toBe(String(eq('count', 5)));
    });

    it('compares booleans as scalars only', () => {
      expect(t({ active: true })).toBe(String(m('active').eq(true)));
    });

    it('matches null only when the key is present', () => {
      expect(t({ price: null })).toBe(String(keys().contains('price').and(m('price').isNull())));
    });

    it('turns dates into ISO strings', () => {
      const date = new Date('2024-01-02T03:04:05.000Z');
      expect(t({ createdAt: { $gte: date } })).toBe(String(m('createdAt').gte('2024-01-02T03:04:05.000Z')));
    });

    it('treats -0 as 0', () => {
      expect(t({ n: { $gt: -0 } })).toBe(String(m('n').gt(0)));
    });

    it('rejects non-finite numbers', () => {
      expect(() => t({ n: { $gt: Number.NaN } })).toThrow(/finite/);
    });
  });

  describe('negation', () => {
    it('$ne matches documents where the field has another value', () => {
      expect(t({ category: { $ne: 'news' } })).toBe(
        String(
          m('category')
            .isNotNull()
            .and(not(eq('category', 'news'))),
        ),
      );
    });

    it('$ne null means the field has a value', () => {
      expect(t({ price: { $ne: null } })).toBe(String(m('price').isNotNull()));
    });

    it('field-level $not negates the inner condition, for documents with the field', () => {
      expect(t({ price: { $not: { $gt: 50 } } })).toBe(
        String(
          m('price')
            .isNotNull()
            .and(not(m('price').gt(50))),
        ),
      );
    });

    it('field-level $not negates a $regex', () => {
      expect(t({ name: { $not: { $regex: '^a', $options: 'i' } } })).toBe(
        String(
          m('name')
            .isNotNull()
            .and(not(m('name').regexpMatch('^a', 'i'))),
        ),
      );
    });

    it('field-level $not of $exists is about the key itself', () => {
      expect(t({ name: { $not: { $exists: true } } })).toBe(String(not(keys().contains('name'))));
    });
  });

  describe('comparison', () => {
    it.each([
      ['$gt', 'gt'],
      ['$gte', 'gte'],
      ['$lt', 'lt'],
      ['$lte', 'lte'],
    ] as const)('translates %s', (op, method) => {
      expect(t({ price: { [op]: 10 } })).toBe(String(m('price')[method](10)));
    });

    it('combines several operators on one field with all()', () => {
      expect(t({ price: { $gte: 10, $lte: 20 } })).toBe(String(all([m('price').gte(10), m('price').lte(20)])));
    });

    it('rejects non-comparable values', () => {
      expect(() => t({ price: { $gt: true as any } })).toThrow(/number, string or Date/);
    });
  });

  describe('$in / $nin', () => {
    it('uses string lists and array membership', () => {
      expect(t({ tag: { $in: ['a', 'b'] } })).toBe(
        String(
          any([m('tag').in(['a', 'b']), isArray('tag').and(any([m('tag').contains('a'), m('tag').contains('b')]))]),
        ),
      );
    });

    it('sends numbers as an f64 list, so ints mixed with floats still match', () => {
      expect(t({ n: { $in: [1, 2.5] } })).toBe(
        String(any([m('n').in(f64List([1, 2.5])), isArray('n').and(any([m('n').contains(1), m('n').contains(2.5)]))])),
      );
    });

    it('handles booleans and null in the list', () => {
      expect(t({ x: { $in: [true, null] } })).toBe(
        String(any([m('x').eq(true), keys().contains('x').and(m('x').isNull())])),
      );
    });

    it('treats a direct array value as $in', () => {
      expect(t({ tag: ['a'] })).toBe(String(any([m('tag').in(['a']), isArray('tag').and(m('tag').contains('a'))])));
    });

    it('matches nothing for an empty $in, everything for an empty $nin', () => {
      expect(t({ tag: { $in: [] } })).toBe(String(literal(false)));
      expect(t({ tag: { $nin: [] } })).toBe(
        String(
          m('tag')
            .isNotNull()
            .and(not(literal(false))),
        ),
      );
    });

    it('$nin negates $in', () => {
      expect(t({ tag: { $nin: ['a'] } })).toBe(
        String(
          m('tag')
            .isNotNull()
            .and(not(any([m('tag').in(['a']), isArray('tag').and(m('tag').contains('a'))]))),
        ),
      );
    });

    it('rejects a non-array', () => {
      expect(() => t({ tag: { $in: 'a' as any } })).toThrow(/array/);
    });
  });

  describe('array operators', () => {
    it('$all requires every value in an array field', () => {
      expect(t({ tags: { $all: ['a', 'b'] } })).toBe(
        String(isArray('tags').and(all([m('tags').contains('a'), m('tags').contains('b')]))),
      );
    });

    it('$all with no values matches nothing', () => {
      expect(t({ tags: { $all: [] } })).toBe(String(literal(false)));
    });

    it('$size compares the stored array length', () => {
      expect(t({ tags: { $size: 2 } })).toBe(String(field('metadata_sizes.tags').eq(2)));
    });

    it('$contains matches substrings and array elements', () => {
      expect(t({ description: { $contains: 'novel' } })).toBe(String(m('description').contains('novel')));
    });

    it('$elemMatch is not supported', () => {
      expect(() => t({ reviews: { $elemMatch: { score: { $gt: 4 } } } } as any)).toThrow(/Unsupported operator/);
    });
  });

  describe('$exists and $regex', () => {
    it('$exists checks the stored key paths', () => {
      expect(t({ description: { $exists: true } })).toBe(String(keys().contains('description')));
      expect(t({ description: { $exists: false } })).toBe(String(not(keys().contains('description'))));
    });

    it('$regex with $options', () => {
      expect(t({ name: { $regex: '^prod', $options: 'i' } })).toBe(String(m('name').regexpMatch('^prod', 'i')));
    });

    it('$regex with a RegExp keeps its flags', () => {
      expect(t({ name: { $regex: /^prod/i } })).toBe(String(m('name').regexpMatch('^prod', 'i')));
    });

    it('a direct RegExp value is a regex match', () => {
      expect(t({ name: /x$/ })).toBe(String(m('name').regexpMatch('x$')));
    });
  });

  describe('logical operators', () => {
    it('combines field conditions with all()', () => {
      expect(t({ a: 'x', b: { $gt: 1 } })).toBe(String(all([eq('a', 'x'), m('b').gt(1)])));
    });

    it('translates $and, $or, $nor and $not', () => {
      expect(t({ $and: [{ a: 'x' }, { b: 'y' }] })).toBe(String(all([eq('a', 'x'), eq('b', 'y')])));
      expect(t({ $or: [{ a: 'x' }, { b: 'y' }] })).toBe(String(any([eq('a', 'x'), eq('b', 'y')])));
      expect(t({ $nor: [{ a: 'x' }, { b: 'y' }] })).toBe(String(not(any([eq('a', 'x'), eq('b', 'y')]))));
      expect(t({ $not: { a: 'x' } })).toBe(String(not(eq('a', 'x'))));
    });

    it('handles empty logical operators like MongoDB', () => {
      expect(t({ $and: [] })).toBe(String(literal(true)));
      expect(t({ $or: [] })).toBe(String(literal(false)));
      expect(t({ $nor: [] })).toBe(String(literal(true)));
    });

    it('rejects an empty $not', () => {
      expect(() => t({ $not: {} })).toThrow(/\$not/);
    });

    it('rejects operators at the top level', () => {
      expect(() => t({ $gt: 1 } as any)).toThrow();
    });

    it('rejects a logical operator without an array', () => {
      expect(() => t({ $and: 'x' } as any)).toThrow();
    });
  });

  describe('nested fields', () => {
    it('uses dot paths for nested objects', () => {
      expect(t({ user: { name: 'ada' } })).toBe(String(eq('user.name', 'ada')));
      expect(t({ 'user.age': { $gte: 18 } })).toBe(String(m('user.age').gte(18)));
    });

    it('rejects a mix of operators and nested fields', () => {
      expect(() => t({ user: { $eq: 1, name: 'x' } } as any)).toThrow(/mixes operators/);
    });
  });
});
