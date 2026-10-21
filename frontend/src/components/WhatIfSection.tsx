import React from 'react'
import { useCopy } from '../context/CopyContext'
import type { components } from '../api/types'

type Counterfactual = components['schemas']['Counterfactual']

interface WhatIfSectionProps {
  counterfactuals?: Counterfactual[]
}

export const WhatIfSection: React.FC<WhatIfSectionProps> = ({ counterfactuals = [] }) => {
  const { t } = useCopy()

  if (!counterfactuals || counterfactuals.length === 0) {
    return null
  }

  const getVerdictBadge = (verdict: string) => {
    const v = verdict.toLowerCase()
    if (v.includes('contra')) {
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300">
          {t.verdicts.badgeContradicted}
        </span>
      )
    }
    if (v.includes('verif') || v.includes('supp')) {
      return (
        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
          {t.verdicts.badgeVerified}
        </span>
      )
    }
    return (
      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-200 text-slate-800 dark:bg-slate-700 dark:text-slate-300">
        {t.verdicts.badgeInsufficient}
      </span>
    )
  }

  return (
    <div className="space-y-3">
      <div>
        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center space-x-1.5">
          <svg className="w-4 h-4 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
          </svg>
          <span>{t.reportPage.whatIfTitle}</span>
        </h4>
        <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
          {t.reportPage.whatIfSubtitle}
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {counterfactuals.map((cf, idx) => (
          <div
            key={idx}
            className={`p-3 rounded-lg border text-xs flex flex-col justify-between transition-colors ${
              cf.flipped
                ? 'bg-purple-50/60 border-purple-200 dark:bg-purple-950/20 dark:border-purple-800/50'
                : 'bg-slate-50 dark:bg-slate-800/40 border-slate-200 dark:border-slate-800'
            }`}
          >
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-purple-900 dark:text-purple-300 text-[11px] uppercase tracking-wide">
                  Scenario #{idx + 1}
                </span>
                {cf.flipped ? (
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-purple-200 text-purple-900 dark:bg-purple-900 dark:text-purple-200">
                    Flipped Verdict
                  </span>
                ) : (
                  <span className="text-[10px] text-slate-400 font-medium">Unchanged</span>
                )}
              </div>
              <p className="text-slate-800 dark:text-slate-200 mb-3 italic">
                "{cf.change_description}"
              </p>
            </div>

            <div className="pt-2 border-t border-slate-200/60 dark:border-slate-700/60 flex items-center justify-between">
              <span className="text-[11px] text-slate-500">
                {cf.flipped ? t.reportPage.verdictFlipped : t.reportPage.verdictNotFlipped}
              </span>
              {cf.flipped && getVerdictBadge(cf.new_verdict)}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
