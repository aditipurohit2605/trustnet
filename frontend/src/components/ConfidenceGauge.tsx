import React, { useState } from 'react'
import { useCopy } from '../context/CopyContext'

interface ConfidenceGaugeProps {
  confidence: number // 0 to 1
}

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({ confidence }) => {
  const { t } = useCopy()
  const [showTooltip, setShowTooltip] = useState(false)

  const pct = Math.round(Math.max(0, Math.min(1, confidence)) * 100)

  // Color tone based on confidence level (avoiding misleading green-for-truth associations)
  let barColor = 'bg-indigo-600 dark:bg-indigo-500'
  if (pct < 50) {
    barColor = 'bg-slate-400 dark:bg-slate-500'
  } else if (pct < 75) {
    barColor = 'bg-sky-600 dark:bg-sky-500'
  }

  return (
    <div className="flex flex-col space-y-1.5 min-w-[140px]">
      <div className="flex items-center justify-between">
        <div className="relative inline-flex items-center space-x-1">
          <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
            {t.confidence.label}
          </span>
          <button
            type="button"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
            onFocus={() => setShowTooltip(true)}
            onBlur={() => setShowTooltip(false)}
            className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 focus:outline-none focus-visible:ring-1 focus-visible:ring-indigo-500 rounded-full"
            aria-label={t.confidence.tooltip}
          >
            <svg
              className="w-3.5 h-3.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M12 18h.007v.008H12V18z"
              />
            </svg>
          </button>

          {/* Accessible Tooltip */}
          {showTooltip && (
            <div
              role="tooltip"
              className="absolute bottom-full left-0 mb-2 w-64 p-2.5 bg-slate-900 text-white text-[11px] leading-relaxed rounded-lg shadow-xl z-50 border border-slate-700 pointer-events-none"
            >
              {t.confidence.tooltip}
              <div className="absolute top-full left-4 -mt-1 border-4 border-transparent border-t-slate-900" />
            </div>
          )}
        </div>

        {/* Numeric percentage strictly without green checkmark */}
        <span className="text-xs font-mono font-bold text-slate-800 dark:text-slate-200">
          {pct}%
        </span>
      </div>

      {/* Progress track */}
      <div
        className="w-full bg-slate-200 dark:bg-slate-700 h-2 rounded-full overflow-hidden"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${t.confidence.label}: ${pct}%`}
      >
        <div
          className={`h-full rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
