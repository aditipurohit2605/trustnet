import React from 'react'
import { useCopy } from '../context/CopyContext'
import type { components } from '../api/types'

type TemporalInfo = components['schemas']['TemporalInfo']

interface TemporalInfoSectionProps {
  temporal?: TemporalInfo | null
}

export const TemporalInfoSection: React.FC<TemporalInfoSectionProps> = ({ temporal }) => {
  const { t } = useCopy()

  if (!temporal) return null

  const { is_stale, claim_time_expressions, newest_evidence_date, oldest_evidence_date, explanation } =
    temporal

  const hasExpressions = claim_time_expressions && claim_time_expressions.length > 0
  const hasTimeline = newest_evidence_date || oldest_evidence_date

  if (!is_stale && !hasExpressions && !hasTimeline && !explanation) {
    return null
  }

  return (
    <div className="space-y-3">
      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center space-x-1.5">
        <svg className="w-4 h-4 text-sky-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span>{t.reportPage.temporalTitle}</span>
      </h4>

      {/* Stale Warning Banner */}
      {is_stale && (
        <div
          role="alert"
          className="p-3 rounded-lg border border-amber-300 bg-amber-50 dark:border-amber-700 dark:bg-amber-950/40 text-xs flex items-start space-x-2.5"
        >
          <svg className="w-4 h-4 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <div className="text-amber-900 dark:text-amber-200">
            <span className="font-semibold">{t.reportPage.temporalStaleWarning}</span>
            {explanation && <p className="mt-1 text-amber-800 dark:text-amber-300">{explanation}</p>}
          </div>
        </div>
      )}

      {!is_stale && explanation && (
        <p className="text-xs text-slate-600 dark:text-slate-400">{explanation}</p>
      )}

      {/* Time expressions & Timeline */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
        {hasExpressions && (
          <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-800">
            <span className="font-medium text-slate-500 dark:text-slate-400 block mb-1">
              {t.reportPage.temporalTimeExpressions}:
            </span>
            <div className="flex flex-wrap gap-1">
              {claim_time_expressions.map((expr, i) => (
                <span
                  key={i}
                  className="px-2 py-0.5 rounded bg-slate-200 dark:bg-slate-700 font-mono text-[11px] text-slate-800 dark:text-slate-200"
                >
                  {expr}
                </span>
              ))}
            </div>
          </div>
        )}

        {hasTimeline && (
          <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-800">
            <span className="font-medium text-slate-500 dark:text-slate-400 block mb-1">
              {t.reportPage.temporalEvidenceRange}:
            </span>
            <span className="font-mono text-slate-800 dark:text-slate-200">
              {oldest_evidence_date || 'Unknown'} — {newest_evidence_date || 'Present'}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
