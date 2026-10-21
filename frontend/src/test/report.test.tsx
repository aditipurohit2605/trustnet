import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { CopyProvider } from '../context/CopyContext'
import { ResearchModeProvider } from '../context/ResearchModeContext'
import { ReportPage } from '../pages/ReportPage'
import { enCopy } from '../copy/en'

function renderReportPage(reportId: string, initialResearchMode = false) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })

  if (initialResearchMode) {
    localStorage.setItem('trustnet_research_mode', 'true')
  }

  return render(
    <QueryClientProvider client={queryClient}>
      <CopyProvider>
        <ResearchModeProvider>
          <MemoryRouter initialEntries={[`/reports/${reportId}`]}>
            <Routes>
              <Route path="/reports/:id" element={<ReportPage />} />
            </Routes>
          </MemoryRouter>
        </ResearchModeProvider>
      </CopyProvider>
    </QueryClientProvider>
  )
}

describe('ReportPage Epistemic Rigor & Verdict Copy Integrity', () => {
  it('renders verified claim with verbatim copy', async () => {
    renderReportPage('rep-verified-123')

    await waitFor(() => {
      // Verbatim wording
      expect(
        screen.getByText(enCopy.verdicts.verified)
      ).toBeInTheDocument()
      expect(
        screen.getByText(enCopy.shareAdvice.safe_to_share)
      ).toBeInTheDocument()
    })

    // Assert strictly no instances of forbidden words as verdicts
    expect(screen.queryByText(/^true$/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/^false$/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/^fake$/i)).not.toBeInTheDocument()
  })

  it('renders abstained claim with special layout, reason, and gap checklist', async () => {
    renderReportPage('rep-abstained-456')

    await waitFor(() => {
      // Verbatim wording for unconfirmed / abstained
      expect(
        screen.getAllByText(enCopy.verdicts.abstained).length
      ).toBeGreaterThan(0)

      // Reason
      expect(
        screen.getByText(/low_confidence_and_out_of_domain/i)
      ).toBeInTheDocument()

      // Gaps checklist
      expect(screen.getByText(/GAP_NO_CHEMICAL_TEST/i)).toBeInTheDocument()

      // Degraded warning banner
      expect(screen.getByText(/Degraded Pipeline Alert/i)).toBeInTheDocument()
      expect(screen.getByText('image_processing')).toBeInTheDocument()
    })
  })

  it('renders print and download actions', async () => {
    const printSpy = vi.spyOn(window, 'print')
    const user = userEvent.setup()
    renderReportPage('rep-verified-123')

    await waitFor(() => {
      expect(screen.getByLabelText(/Print Report/i)).toBeInTheDocument()
    })

    const printBtn = screen.getByLabelText(/Print Report/i)
    await user.click(printBtn)
    expect(printSpy).toHaveBeenCalled()

    printSpy.mockRestore()
  })

  it('renders Research Mode inspector when enabled', async () => {
    renderReportPage('rep-verified-123', true)

    await waitFor(() => {
      expect(screen.getByText(/Research & Calibration Inspector/i)).toBeInTheDocument()
      expect(screen.getByText(/Pipeline Version/i)).toBeInTheDocument()
      expect(screen.getByText(/Stage Latency Breakdown/i)).toBeInTheDocument()
    })
  })

  it('renders evidence without executing injected script tags (XSS safety)', async () => {
    renderReportPage('rep-verified-123')

    await waitFor(() => {
      expect(screen.getByText(/WHO confirms booster shot reduces hospitalization/i)).toBeInTheDocument()
    })

    // Verify no unexpected alert or raw unescaped script elements in the DOM
    const scripts = document.querySelectorAll('script')
    for (const s of scripts) {
      expect(s.textContent).not.toContain('alert(1)')
    }
  })
})
