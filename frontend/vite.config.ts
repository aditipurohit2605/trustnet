import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vitest/config'

/**
 * Dev server mock plugin for /v1/version endpoint.
 * Ensures client.ts can probe version metadata without modifying backend API code.
 */
function versionDevPlugin(): Plugin {
  return {
    name: 'trustnet-version-dev-plugin',
    configureServer(server) {
      server.middlewares.use('/v1/version', (_req, res) => {
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ version: '0.1.0', service: 'trustnet-api', status: 'operational' }))
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), versionDevPlugin()],
  server: {
    port: 5173,
    headers: {
      // Production-grade security headers without unsafe-inline scripts
      'Content-Security-Policy': [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com data:",
        "img-src 'self' data: blob:",
        "connect-src 'self' http://localhost:8000 http://127.0.0.1:8000",
        "frame-ancestors 'none'",
      ].join('; '),
      'X-Frame-Options': 'DENY',
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'strict-origin-when-cross-origin',
    },
    proxy: {
      '/v1': {
        target: process.env.TRUSTNET_API_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
        configure: (proxy) => {
          // Attaches the API key strictly on the Node.js server side.
          // This key NEVER appears in client bundles or any VITE_ environment variable.
          proxy.on('proxyReq', (proxyReq) => {
            const rawKeys = process.env.TRUSTNET_API_KEY || process.env.TRUSTNET_API_KEYS
            if (rawKeys) {
              const primaryKey = rawKeys.split(',')[0].trim()
              if (primaryKey) {
                proxyReq.setHeader('X-API-Key', primaryKey)
              }
            }
          })
        },
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
})
