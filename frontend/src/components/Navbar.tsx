import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useCopy } from '../context/CopyContext'
import { useResearchMode } from '../context/ResearchModeContext'

export const Navbar: React.FC = () => {
  const { lang, setLang, t } = useCopy()
  const { researchMode, toggleResearchMode } = useResearchMode()
  const location = useLocation()

  const isActive = (path: string) => {
    if (path === '/' && location.pathname === '/') return true
    if (path !== '/' && location.pathname.startsWith(path)) return true
    return false
  }

  return (
    <>
      {/* Accessibility: Skip link */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-indigo-600 focus:text-white focus:rounded-md focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-white"
      >
        Skip to main content
      </a>

      <header className="sticky top-0 z-40 w-full border-b border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md transition-colors print:hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Brand Logo & Name */}
            <div className="flex items-center space-x-3">
              <Link
                to="/"
                className="flex items-center space-x-2.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 rounded-lg p-1"
                aria-label="TrustNet Home"
              >
                <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-white shadow-md shadow-indigo-500/20">
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2.2}
                    aria-hidden="true"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z"
                    />
                  </svg>
                </div>
                <div className="flex flex-col">
                  <span className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">
                    {t.common.appName}
                  </span>
                  <span className="text-[10px] text-slate-500 dark:text-slate-400 font-medium tracking-wide hidden sm:inline">
                    Epistemic Verification
                  </span>
                </div>
              </Link>
            </div>

            {/* Navigation Tabs */}
            <nav className="flex items-center space-x-1 sm:space-x-2" aria-label="Main Navigation">
              <Link
                to="/"
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                  isActive('/')
                    ? 'bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400'
                    : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800'
                }`}
              >
                {t.nav.check}
              </Link>
              <Link
                to="/compare"
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                  isActive('/compare')
                    ? 'bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400'
                    : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800'
                }`}
              >
                {t.nav.compare}
              </Link>
              <Link
                to="/about"
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                  isActive('/about')
                    ? 'bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400'
                    : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800'
                }`}
              >
                {t.nav.about}
              </Link>
            </nav>

            {/* Controls: Research Mode & Language Selector */}
            <div className="flex items-center space-x-2.5">
              {/* Research Mode Toggle */}
              <button
                type="button"
                onClick={toggleResearchMode}
                className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 border ${
                  researchMode
                    ? 'bg-purple-100 text-purple-900 border-purple-300 dark:bg-purple-950 dark:text-purple-200 dark:border-purple-700 shadow-sm'
                    : 'bg-slate-100 text-slate-600 border-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:border-slate-700 hover:bg-slate-200 dark:hover:bg-slate-700'
                }`}
                title="Toggle Research / Calibration Inspector mode"
                aria-pressed={researchMode}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    researchMode ? 'bg-purple-500 animate-pulse' : 'bg-slate-400'
                  }`}
                  aria-hidden="true"
                />
                <span className="hidden md:inline">{t.common.researchMode}</span>
                <span className="md:hidden">Lab</span>
              </button>

              {/* Language Switcher */}
              <div className="inline-flex rounded-lg border border-slate-200 dark:border-slate-700 p-0.5 bg-slate-50 dark:bg-slate-800">
                <button
                  type="button"
                  onClick={() => setLang('en')}
                  className={`px-2 py-0.5 text-xs font-medium rounded-md transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-indigo-500 ${
                    lang === 'en'
                      ? 'bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-xs font-semibold'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                  }`}
                  aria-pressed={lang === 'en'}
                >
                  EN
                </button>
                <button
                  type="button"
                  onClick={() => setLang('hi')}
                  className={`px-2 py-0.5 text-xs font-medium rounded-md transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-indigo-500 ${
                    lang === 'hi'
                      ? 'bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-xs font-semibold'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                  }`}
                  aria-pressed={lang === 'hi'}
                >
                  हिन्दी
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>
    </>
  )
}
