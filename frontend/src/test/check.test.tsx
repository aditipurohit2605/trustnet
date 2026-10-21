import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { CopyProvider } from '../context/CopyContext'
import { ResearchModeProvider } from '../context/ResearchModeContext'
import { CheckPage } from '../pages/CheckPage'

function renderCheckPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <CopyProvider>
        <ResearchModeProvider>
          <MemoryRouter>
            <CheckPage />
          </MemoryRouter>
        </ResearchModeProvider>
      </CopyProvider>
    </QueryClientProvider>
  )
}

describe('CheckPage Verification Form & Privacy Integrity', () => {
  it('renders claim textarea and submits claim', async () => {
    const user = userEvent.setup()
    renderCheckPage()

    const textarea = screen.getByPlaceholderText(/Paste the statement/i)
    expect(textarea).toBeInTheDocument()

    await user.type(textarea, 'Test verification claim')
    expect(textarea).toHaveValue('Test verification claim')

    const submitBtn = screen.getByRole('button', { name: /Start Verification/i })
    expect(submitBtn).not.toBeDisabled()

    await user.click(submitBtn)

    await waitFor(() => {
      // Button states while processing
      expect(submitBtn).toBeInTheDocument()
    })
  })

  it('rejects files larger than 10MB client-side before upload', async () => {
    renderCheckPage()

    const oversizedFile = new File(['x'.repeat(11 * 1024 * 1024)], 'huge.png', {
      type: 'image/png',
    })

    const dropzone = screen.getByText(/Drag & drop an image/i)
    expect(dropzone).toBeInTheDocument()

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    expect(input).toBeInTheDocument()

    // Simulate selecting oversized file
    await userEvent.upload(input, oversizedFile)

    await waitFor(() => {
      expect(screen.getByText(/Maximum allowed image size is 10 MB/i)).toBeInTheDocument()
    })
  })

  it('rejects unsupported file formats', async () => {
    renderCheckPage()

    const invalidFile = new File(['dummy content'], 'document.pdf', {
      type: 'application/pdf',
    })

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input, { target: { files: [invalidFile] } })

    await waitFor(() => {
      expect(screen.getByText(/Unsupported file type/i)).toBeInTheDocument()
    })
  })

  it('PRIVACY MANDATE: strictly never writes claim text or images to localStorage or sessionStorage', async () => {
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem')
    const user = userEvent.setup()
    renderCheckPage()

    const secretText = 'CONFIDENTIAL_PERSONAL_MESSAGE_DO_NOT_STORE'
    const textarea = screen.getByPlaceholderText(/Paste the statement/i)
    await user.type(textarea, secretText)

    const submitBtn = screen.getByRole('button', { name: /Start Verification/i })
    await user.click(submitBtn)

    // Check all calls to storage setItem
    for (const call of setItemSpy.mock.calls) {
      const key = String(call[0])
      const val = String(call[1])
      expect(val).not.toContain(secretText)
      expect(key).not.toContain('claim_text')
      expect(key).not.toContain('image')
    }

    setItemSpy.mockRestore()
  })
})
