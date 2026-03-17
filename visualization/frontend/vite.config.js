import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],

  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws':  { target: 'ws://localhost:8000',  ws: true },
    },
  },

  // Molstar ships CommonJS + ESM mixed bundles — exclude from pre-bundling
  optimizeDeps: {
    exclude: ['molstar'],
  },

  // Molstar uses BigInt, top-level await, and dynamic imports
  build: {
    target: 'esnext',
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          molstar: ['molstar'],
        },
      },
    },
  },

  worker: {
    format: 'es',
  },
})
