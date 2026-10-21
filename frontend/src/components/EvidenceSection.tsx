import React, { useState } from 'react'
import { useCopy } from '../context/CopyContext'
import type { components } from '../api/types'

type Evidence = components['schemas']['Evidence']
type ConflictInfo = components['schemas']['ConflictInfo']
type SourceWeight = components['schemas']['SourceWeight']

interface EvidenceSectionProps {
  supporting: Evidence[]
  contradicting: Evidence[]
  conflict?: ConflictInfo | null
  sourceWeights?: SourceWeight[]
}

export const EvidenceSection: React.FC<EvidenceSectionProps> = ({
  supporting,
  contradicting,
  conflict,
  sourceWeights = [],
}) => {
  const { t } = useCopy()
  const [activeWeightTooltip, setActiveWeightTooltip] = useState<string | null>(null)

  const weightMap = new Map<string, SourceWeight>()
  for (const sw of sourceWeights) {
    if (sw.source_type) {
      weightMap.set(sw.source_type.toLowerCase(), sw)
    }
  }

  const getSourceBadge = (sourceType?: string | null) => {
    const type = (sourceType || 'user').toLowerCase()
    let label = t.reportPage.sourceTypeUser
    let color = 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-300 border-slate-300 dark:border-slate-700'

    if (type.includes('factcheck')) {
      label = t.reportPage.sourceTypeFactcheck
      color = 'bg-emerald-50 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700'
    } else if (type.includes('news')) {
      label = t.reportPage.sourceTypeNews
      color = 'bg-blue-50 text-blue-800 dark:bg-blue-950/60 dark:text-blue-300 border-blue-300 dark:border-blue-700'
    } else if (type.includes('academic')) {
      label = t.reportPage.sourceTypeAcademic
      color = 'bg-purple-50 text-purple-800 dark:bg-purple-950/60 dark:text-purple-300 border-purple-300 dark:border-purple-700'
    }

    const sw = weightMap.get(type)

    return (
      <div className="relative inline-flex items-center space-x-1.5">
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold border ${color}`}>
          {label}
        </span>
        {sw && (
          <div className="relative inline-block">
            <button
              type="button"
              onMouseEnter={() => setActiveWeightTooltip(type)}
              onMouseLeave={() => setActiveWeightTooltip(null)}
              onFocus={() => setActiveWeightTooltip(type)}
              onBlur={() => setActiveWeightTooltip(null)}
              className="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-300 focus:outline-none"
              aria-label={`Source Weight: ${sw.weight}. ${sw.rationale}`}
            >
              w: {sw.weight.toFixed(2)}
            </button>
            {activeWeightTooltip === type && (
              <div
                role="tooltip"
                className="absolute bottom-full left-0 mb-1.5 w-56 p-2 bg-slate-900 text-white text-[10px] leading-snug rounded shadow-lg z-50 pointer-events-none"
              >
                <div className="font-semibold text-indigo-300 mb-0.5">
                  {sw.source_type} (Weight: {sw.weight})
                </div>
                <div>{sw.rationale}</div>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  const renderCard = (ev: Evidence, index: number, isSupporting: boolean) => {
    return (
      <div
        key={index}
        className={`rounded-lg border p-3.5 text-xs transition-shadow ${
          isSupporting
            ? 'border-emerald-200 bg-emerald-50/40 dark:border-emerald-800/50 dark:bg-emerald-950/20'
            : 'border-rose-200 bg-rose-50/40 dark:border-rose-800/50 dark:bg-rose-950/20'
        }`}
      >
        <div className="flex items-center justify-between mb-2">
          {getSourceBadge(ev.source_type)}
          {ev.date && (
            <span className="text-[11px] text-slate-500 dark:text-slate-400 font-mono">
              {ev.date}
            </span>
          )}
        </div>

        <p className="text-slate-800 dark:text-slate-200 leading-relaxed font-sans mb-2.5">
          "{ev.text}"
        </p>

        {ev.url && (
          <div className="flex items-center justify-between pt-1 border-t border-slate-200/60 dark:border-slate-800/60">
            <a
              href={ev.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center space-x-1 text-[11px] font-medium text-indigo-600 dark:text-indigo-400 hover:underline"
            >
              <span>{t.reportPage.viewSource}</span>
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
            {ev.score != null && (
              <span className="text-[10px] text-slate-500 font-mono">
                rel: {ev.score.toFixed(2)}
              </span>
            )}
          </div>
        )}
      </div>
    )
  }

  const hasConflict = conflict?.has_conflict

  return (
    <div className="space-y-4">
      {/* Evidence Conflict Notification */}
      {hasConflict && (
        <div
          role="alert"
          className="rounded-xl border border-amber-300 bg-amber-50 dark:border-amber-700 dark:bg-amber-950/40 p-4"
        >
          <div className="flex items-start space-x-3">
            <svg
              className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
              />
            </svg>
            <div>
              <h4 className="text-sm font-semibold text-amber-900 dark:text-amber-200">
                {t.reportPage.conflictTitle}
              </h4>
              <p className="mt-1 text-xs text-amber-800 dark:text-amber-300">
                {conflict.explanation || t.reportPage.conflictExplanation}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Side-by-side or stacked layout */}
      <div className={`grid gap-4 ${hasConflict ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 md:grid-cols-2'}`}>
        {/* Supporting Evidence */}
        <div className="space-y-2.5">
          <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-800 dark:text-emerald-400 flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>{t.reportPage.supportingSources.replace('{count}', String(supporting.length))}</span>
          </h4>
          {supporting.length === 0 ? (
            <p className="text-xs text-slate-500 italic p-3 rounded-lg bg-slate-50 dark:bg-slate-800/40">
              {t.reportPage.noSupportingEvidence}
            </p>
          ) : (
            supporting.map((ev, i) => renderCard(ev, i, true))
          )}
        </div>

        {/* Contradicting Evidence */}
        <div className="space-y-2.5">
          <h4 className="text-xs font-bold uppercase tracking-wider text-rose-800 dark:text-rose-400 flex items-center space-x-1.5">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            <span>{t.reportPage.contradictingSources.replace('{count}', String(contradicting.length))}</span>
          </h4>
          {contradicting.length === 0 ? (
            <p className="text-xs text-slate-500 italic p-3 rounded-lg bg-slate-50 dark:bg-slate-800/40">
              {t.reportPage.noContradictingEvidence}
            </p>
          ) : (
            contradicting.map((ev, i) => renderCard(ev, i, false))
          )}
        </div>
      </div>
    </div>
  )
}
