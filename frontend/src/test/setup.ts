import '@testing-library/jest-dom'
import { afterAll, afterEach, beforeAll } from 'vitest'
import { cleanup } from '@testing-library/react'
import { setupServer } from 'msw/node'
import { handlers } from './handlers'

export const server = setupServer(...handlers)

beforeAll(() => {
  server.listen()
})

afterEach(() => {
  cleanup()
  server.resetHandlers()
  localStorage.clear()
  sessionStorage.clear()
})

afterAll(() => {
  server.close()
})

// Mock window.print
Object.defineProperty(window, 'print', {
  value: () => {},
  writable: true,
})

// Mock URL.createObjectURL and URL.revokeObjectURL
if (typeof window.URL.createObjectURL === 'undefined') {
  Object.defineProperty(window.URL, 'createObjectURL', {
    value: () => 'blob:mock-image-url',
    writable: true,
  })
}
if (typeof window.URL.revokeObjectURL === 'undefined') {
  Object.defineProperty(window.URL, 'revokeObjectURL', {
    value: () => {},
    writable: true,
  })
}
