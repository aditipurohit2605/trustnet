import React from 'react'
import { useCopy } from '../context/CopyContext'
import type { components } from '../api/types'

type ManipulationTag = components['schemas']['ManipulationTag']

interface StyleSignalsSectionProps {
  signals?: ManipulationTag[]
}

export const StyleSignalsSection: React.FC<StyleSignalsSectionProps> = ({ signals = [] }) => {
  const { t } = useCopy()

  if (!signals || signals.length === 0) {
    return null
  }

  const formatTagName = (name: string) => {
    return name
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase())
  }

  return (
    <div className="space-y-3 pt-2">
      <div>
        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center space-x-1.5">
          <svg className="w-4 h-4 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
          </svg>
          <span>{t.reportPage.styleSignalsTitle}</span>
        </h4>
        
        {/* Prominent Disclaimer */}
        <div className="mt-1.5 p-2 rounded-md bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800/40 text-[11px] text-amber-900 dark:text-amber-200 font-medium">
          ⚠️ {t.reportPage.styleSignalsDisclaimer}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
        {signals.map((sig, idx) => {
          const pct = Math.round((sig.confidence || 0) * 100)
          return (
            <div
              key={idx}
              className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-xs shadow-xs"
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-semibold text-slate-900 dark:text-slate-100">
                  {formatTagName(sig.name)}
                </span>
                <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-300">
                  {pct}%
                </span>
              </div>

              {sig.span && (
                <div className="mt-1 text-[11px] text-slate-600 dark:text-slate-400 italic bg-slate-50 dark:bg-slate-800/50 p-2 rounded border border-slate-100 dark:border-slate-800">
                  "{sig.span}"
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
