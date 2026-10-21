import React from 'react'
import { Link } from 'react-router-dom'
import { useCopy } from '../context/CopyContext'

export const Footer: React.FC = () => {
  const { t } = useCopy()

  return (
    <footer className="mt-auto border-t border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 py-8 transition-colors print:hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Epistemic disclaimer */}
          <div className="flex flex-col text-center md:text-left space-y-1">
            <p className="text-xs text-slate-600 dark:text-slate-400 font-medium">
              TrustNet is an evidence-grounded verification assistant, not an arbiter of absolute truth.
            </p>
            <p className="text-[11px] text-slate-500 dark:text-slate-500">
              Privacy guarantee: Message texts and images are never stored in your browser storage (localStorage, sessionStorage, IndexedDB, or cookies).
            </p>
          </div>

          {/* Quick links & version */}
          <div className="flex items-center space-x-4 text-xs text-slate-500 dark:text-slate-400">
            <Link
              to="/about"
              className="hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-indigo-500"
            >
              {t.nav.about}
            </Link>
            <span>•</span>
            <span className="font-mono text-[11px] bg-slate-200/70 dark:bg-slate-800 px-2 py-0.5 rounded">
              v0.1.0-stage11
            </span>
          </div>
        </div>
      </div>
    </footer>
  )
}
