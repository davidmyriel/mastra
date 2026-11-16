import { resolve } from 'node:path';
import { config } from 'dotenv';
import { defineConfig } from 'vitest/config';

config();

export default defineConfig({
  test: {
    name: 'e2e:stores/topk',
    environment: 'node',
    include: ['src/**/*.test.ts'],
    // Live tests create TopK collections and wait for writes to be readable.
    testTimeout: 60_000,
    hookTimeout: 120_000,
  },
  resolve: {
    alias: {
      '@mastra/core': resolve(__dirname, '../../packages/core/src'),
    },
  },
});
