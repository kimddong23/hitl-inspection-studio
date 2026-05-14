import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// streamlit custom component build config
// - dev server on 3001 with CORS (Streamlit iframe loads from 8501)
// - relative base for production so dist/index.html works under streamlit's static server
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 3001,
    strictPort: true,
    cors: true,
    headers: {
      'Access-Control-Allow-Origin': '*',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
})
