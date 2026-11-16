import type { MastraDBMessage } from '@mastra/core/agent';
import { InMemoryStore } from '@mastra/core/storage';
import { Memory } from '@mastra/memory';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { TopKVector } from './index';

/**
 * Mastra Memory semantic recall with TopK embedding on the server: `Memory` has no `embedder`,
 * so it sends the message text and TopK embeds it. Runs only when TOPK_API_KEY is set, and
 * deletes only the collections created under this run's prefix.
 */
const apiKey = process.env.TOPK_API_KEY;
const region = process.env.TOPK_REGION ?? 'aws-us-east-1-elastica';
const describeLive = apiKey ? describe : describe.skip;

describeLive('TopKVector Memory Integration Tests', () => {
  const indexPrefix = `mastra-it-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}-memory-`;
  const vector = new TopKVector({ id: 'topk-memory-it', apiKey, region, indexPrefix });
  const threadId = 'thread-1';
  const resourceId = 'user-1';
  let memory: Memory;

  const savedAt = new Date('2026-01-01T00:00:00.000Z');
  const message = (id: string, text: string, thread = threadId, resource = resourceId): MastraDBMessage => ({
    id,
    role: 'user',
    content: { format: 2, parts: [{ type: 'text', text }], content: text },
    createdAt: savedAt,
    threadId: thread,
    resourceId: resource,
  });

  beforeAll(async () => {
    memory = new Memory({
      storage: new InMemoryStore(),
      vector,
      options: {
        semanticRecall: { topK: 1, messageRange: 0, scope: 'resource' },
        // History off, so recall returns only what semantic search selected.
        lastMessages: false,
        generateTitle: false,
      },
    });
    await memory.saveThread({
      thread: { id: threadId, resourceId, title: 'Planning', createdAt: savedAt, updatedAt: savedAt },
    });
    await memory.saveThread({
      thread: { id: 'thread-2', resourceId: 'user-2', title: 'Other user', createdAt: savedAt, updatedAt: savedAt },
    });
    await memory.saveMessages({
      messages: [
        message('msg-deadline', 'We agreed the deadline for the migration is the last Friday of March.'),
        message('msg-lunch', 'The team ordered pizza for the retrospective on Tuesday.'),
        message('msg-budget', 'The hosting budget was raised to cover the new analytics cluster.'),
        message('msg-other', 'My migration deadline is next Monday.', 'thread-2', 'user-2'),
      ],
    });
  }, 180_000);

  afterAll(async () => {
    for (const indexName of await vector.listIndexes()) {
      await vector.deleteIndex({ indexName }).catch(() => {});
    }
  }, 120_000);

  it('stores message text in a collection TopK embeds itself', async () => {
    expect(await vector.listIndexes()).toContain('memory_messages_selfembed');
    const stats = await vector.describeIndex({ indexName: 'memory_messages_selfembed' });
    expect(stats.count).toBe(4);
    expect(stats.dimension).toBe(0);
  });

  it('recalls the semantically matching message without an embedder', async () => {
    const recalled = await memory.recall({ threadId, resourceId, vectorSearchString: 'when is the migration due?' });
    expect(recalled.messages.map(m => m.id)).toEqual(['msg-deadline']);
  });

  it("never recalls another user's messages", async () => {
    const recalled = await memory.recall({
      threadId: 'thread-2',
      resourceId: 'user-2',
      vectorSearchString: 'when is the migration due?',
    });
    expect(recalled.messages.map(m => m.id)).toEqual(['msg-other']);
  });

  it('re-embeds an edited message', async () => {
    const edited = 'The onboarding session moved to the second week of June.';
    await memory.updateMessages({
      messages: [
        { id: 'msg-budget', content: { format: 2, parts: [{ type: 'text', text: edited }], content: edited } } as any,
      ],
    });
    const recalled = await memory.recall({
      threadId,
      resourceId,
      vectorSearchString: 'when is the onboarding session?',
    });
    expect(recalled.messages.map(m => m.id)).toEqual(['msg-budget']);
  });

  it('removes a deleted thread from the index', async () => {
    await memory.deleteThread('thread-2');
    // Memory removes a deleted thread's vectors in the background.
    await memory.settled();
    const rows = await vector.query({
      indexName: 'memory_messages_selfembed',
      filter: { thread_id: 'thread-2' },
      topK: 10,
    });
    expect(rows).toEqual([]);
  });
});
