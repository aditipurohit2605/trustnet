import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { CopyProvider } from '../context/CopyContext'
import { ComparePage } from '../pages/ComparePage'

function renderComparePage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <CopyProvider>
        <MemoryRouter>
          <ComparePage />
        </MemoryRouter>
      </CopyProvider>
    </QueryClientProvider>
  )
}

describe('ComparePage Variant Evaluation Matrix', () => {
  it('renders variant list and epistemic disclaimer', () => {
    renderComparePage()

    expect(screen.getByText(/Variant Comparison Matrix/i)).toBeInTheDocument()
    expect(screen.getByText('text_only')).toBeInTheDocument()
    expect(screen.getByText('text_retrieval')).toBeInTheDocument()
    expect(screen.getByText('text_image')).toBeInTheDocument()
    expect(screen.getByText('full')).toBeInTheDocument()

    expect(
      screen.getByText(/Why do different variants produce different results\?/i)
    ).toBeInTheDocument()
  })

  it('submits multi-variant comparison across all 4 variants', async () => {
    const user = userEvent.setup()
    renderComparePage()

    const input = screen.getByPlaceholderText(/e.g., WHO declared drinking coffee/i)
    await user.type(input, 'Test compare statement')

    const submitBtn = screen.getByRole('button', { name: /Run All 4 Variants Simultaneously/i })
    await user.click(submitBtn)

    await waitFor(() => {
      // After submission, variants should enter running or completed state
      expect(screen.queryByText(/Idle/i)).not.toBeInTheDocument()
    })
  })
})
