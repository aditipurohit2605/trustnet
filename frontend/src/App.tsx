import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { CopyProvider } from './context/CopyContext'
import { ResearchModeProvider } from './context/ResearchModeContext'
import { Navbar } from './components/Navbar'
import { Footer } from './components/Footer'
import { CheckPage } from './pages/CheckPage'
import { ProgressPage } from './pages/ProgressPage'
import { ReportPage } from './pages/ReportPage'
import { ComparePage } from './pages/ComparePage'
import { AboutPage } from './pages/AboutPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 30000,
    },
  },
})

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <CopyProvider>
        <ResearchModeProvider>
          <BrowserRouter>
            <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans antialiased selection:bg-indigo-500 selection:text-white">
              <Navbar />
              <main id="main-content" tabIndex={-1} className="flex-1 focus:outline-none">
                <Routes>
                  <Route path="/" element={<CheckPage />} />
                  <Route path="/jobs/:id" element={<ProgressPage />} />
                  <Route path="/reports/:id" element={<ReportPage />} />
                  <Route path="/compare" element={<ComparePage />} />
                  <Route path="/about" element={<AboutPage />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </main>
              <Footer />
            </div>
          </BrowserRouter>
        </ResearchModeProvider>
      </CopyProvider>
    </QueryClientProvider>
  )
}

export default App
