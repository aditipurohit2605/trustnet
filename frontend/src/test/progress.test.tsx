import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { CopyProvider } from '../context/CopyContext'
import { ProgressPage } from '../pages/ProgressPage'

function renderProgressPage(jobId: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <CopyProvider>
        <MemoryRouter initialEntries={[`/jobs/${jobId}`]}>
          <Routes>
            <Route path="/jobs/:id" element={<ProgressPage />} />
          </Routes>
        </MemoryRouter>
      </CopyProvider>
    </QueryClientProvider>
  )
}

describe('ProgressPage Job Lifecycle & Stage Polling', () => {
  it('renders running stage and progress bar', async () => {
    renderProgressPage('job-running-test')

    await waitFor(() => {
      expect(screen.getByText(/Analyzing Claim/i)).toBeInTheDocument()
      expect(screen.getByText(/evidence retrieval/i)).toBeInTheDocument()
      expect(screen.getByText(/65%/i)).toBeInTheDocument()
    })
  })

  it('renders failed state with error message', async () => {
    renderProgressPage('job-failed')

    await waitFor(() => {
      expect(screen.getByText(/Analysis Failed/i)).toBeInTheDocument()
      expect(screen.getByText(/Failed to complete multimodal grounding/i)).toBeInTheDocument()
    })
  })
})
