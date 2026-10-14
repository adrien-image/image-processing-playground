/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/image-processing-playground/',
  build: {
    // TensorFlow.js (~900 kB) is only in the lazily loaded Classification chunk,
    // so a large chunk there doesn't slow down the first page load.
    chunkSizeWarningLimit: 1000,
  },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
})
