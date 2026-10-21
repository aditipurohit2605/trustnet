import React from 'react'
import { useCopy } from '../context/CopyContext'

interface DegradedBannerProps {
  degraded?: string[]
}

export const DegradedBanner: React.FC<DegradedBannerProps> = ({ degraded }) => {
  const { t } = useCopy()

  if (!degraded || degraded.length === 0) {
    return null
  }

  return (
    <div
      role="alert"
      className="mb-6 rounded-xl border border-amber-300 bg-amber-50 dark:border-amber-700/60 dark:bg-amber-950/40 p-4 shadow-xs"
    >
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 pt-0.5">
          <svg
            className="w-5 h-5 text-amber-600 dark:text-amber-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
            />
          </svg>
        </div>
        <div className="flex-1">
          <h2 className="text-sm font-semibold text-amber-900 dark:text-amber-200">
            {t.reportPage.degradedWarningTitle}
          </h2>
          <p className="mt-1 text-xs text-amber-800 dark:text-amber-300">
            {t.reportPage.degradedWarningDesc}
          </p>
          <ul className="mt-2 flex flex-wrap gap-1.5">
            {degraded.map((item, idx) => (
              <li
                key={idx}
                className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-mono font-medium bg-amber-200/70 text-amber-900 dark:bg-amber-900/60 dark:text-amber-200"
              >
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}
