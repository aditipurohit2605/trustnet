import React, { useState } from 'react'
import { useCopy } from '../context/CopyContext'
import type { components } from '../api/types'

type Gap = components['schemas']['Gap']

interface GapsChecklistProps {
  gaps?: Gap[]
}

export const GapsChecklist: React.FC<GapsChecklistProps> = ({ gaps = [] }) => {
  const { t } = useCopy()
  const [checkedItems, setCheckedItems] = useState<Record<string, boolean>>({})

  if (!gaps || gaps.length === 0) {
    return null
  }

  const toggleCheck = (code: string) => {
    setCheckedItems((prev) => ({
      ...prev,
      [code]: !prev[code],
    }))
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center space-x-1.5">
          <svg className="w-4 h-4 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
          </svg>
          <span>{t.reportPage.gapsTitle}</span>
        </h4>
        <span className="text-[11px] font-medium text-slate-500 bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded">
          {gaps.length} {gaps.length === 1 ? 'gap' : 'gaps'}
        </span>
      </div>

      <p className="text-xs text-slate-600 dark:text-slate-400">
        {t.reportPage.gapsExplanation}
      </p>

      <div className="space-y-2.5">
        {gaps.map((gap) => {
          const isDone = !!checkedItems[gap.code]
          return (
            <div
              key={gap.code}
              className={`p-3 rounded-lg border text-xs transition-colors ${
                isDone
                  ? 'bg-slate-100/70 border-slate-300 dark:bg-slate-800/40 dark:border-slate-700 opacity-75'
                  : 'bg-amber-50/50 border-amber-200/80 dark:bg-amber-950/20 dark:border-amber-800/40'
              }`}
            >
              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id={`gap-${gap.code}`}
                  checked={isDone}
                  onChange={() => toggleCheck(gap.code)}
                  className="mt-0.5 h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 focus:ring-offset-0 cursor-pointer"
                />
                <div className="flex-1">
                  <label
                    htmlFor={`gap-${gap.code}`}
                    className={`font-semibold cursor-pointer ${
                      isDone
                        ? 'line-through text-slate-500 dark:text-slate-400'
                        : 'text-slate-900 dark:text-slate-100'
                    }`}
                  >
                    <span className="font-mono text-[10px] px-1.5 py-0.5 mr-1.5 rounded bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300">
                      {gap.code}
                    </span>
                    {(gap as any).status && (
                      <span className={`text-[9px] uppercase tracking-wider font-semibold px-1.5 py-0.5 mr-2 rounded ${
                        (gap as any).status === 'searched_not_found'
                          ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300'
                          : (gap as any).status === 'outside_scope'
                          ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300'
                          : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300'
                      }`}>
                        {String((gap as any).status).replace(/_/g, ' ')}
                      </span>
                    )}
                    {gap.description}
                  </label>
                  {gap.how_to_check && (
                    <div className="mt-1.5 text-[11px] text-slate-600 dark:text-slate-300 bg-white/70 dark:bg-slate-900/60 p-2 rounded border border-slate-200/60 dark:border-slate-800">
                      <span className="font-medium text-indigo-600 dark:text-indigo-400 mr-1">
                        {t.reportPage.howToCheck}
                      </span>
                      <span>{gap.how_to_check}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
