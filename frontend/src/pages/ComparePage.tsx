import React, { useState, useEffect } from 'react'
import { useCopy } from '../context/CopyContext'
import { verifyClaim, getJob, type PipelineVariant } from '../api/client'
import type { components } from '../api/types'

type ReportEnvelope = components['schemas']['ReportEnvelope']

interface VariantResult {
  variant: PipelineVariant
  status: 'idle' | 'running' | 'done' | 'failed'
  progress: number
  report?: ReportEnvelope
  error?: string
}

export const ComparePage: React.FC = () => {
  const { t } = useCopy()

  const [claimText, setClaimText] = useState('')
  const [results, setResults] = useState<Record<PipelineVariant, VariantResult>>({
    text_only: { variant: 'text_only', status: 'idle', progress: 0 },
    text_retrieval: { variant: 'text_retrieval', status: 'idle', progress: 0 },
    text_image: { variant: 'text_image', status: 'idle', progress: 0 },
    full: { variant: 'full', status: 'idle', progress: 0 },
  })
  const [isRunning, setIsRunning] = useState(false)

  // Track active jobs for polling
  const [activeJobIds, setActiveJobIds] = useState<Record<PipelineVariant, string | null>>({
    text_only: null,
    text_retrieval: null,
    text_image: null,
    full: null,
  })

  // Polling loop
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(async () => {
      let allDone = true

      for (const v of ['text_only', 'text_retrieval', 'text_image', 'full'] as PipelineVariant[]) {
        const jobId = activeJobIds[v]
        if (!jobId) continue
        const cur = results[v]
        if (cur.status === 'done' || cur.status === 'failed') continue

        allDone = false

        try {
          const job = await getJob(jobId)
          if (job.status === 'done' && job.report) {
            setResults((prev) => ({
              ...prev,
              [v]: { ...prev[v], status: 'done', progress: 1, report: job.report },
            }))
          } else if (job.status === 'failed') {
            setResults((prev) => ({
              ...prev,
              [v]: { ...prev[v], status: 'failed', error: job.error || 'Job failed' },
            }))
          } else {
            setResults((prev) => ({
              ...prev,
              [v]: { ...prev[v], progress: job.progress || 0.1 },
            }))
          }
        } catch (err: unknown) {
          const e = err as { message?: string }
          setResults((prev) => ({
            ...prev,
            [v]: { ...prev[v], status: 'failed', error: e.message || 'Polling error' },
          }))
        }
      }

      if (allDone) {
        setIsRunning(false)
      }
    }, 1200)

    return () => clearInterval(interval)
  }, [isRunning, activeJobIds, results])

  const handleStartCompare = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!claimText.trim()) return

    setIsRunning(true)
    const variants: PipelineVariant[] = ['text_only', 'text_retrieval', 'text_image', 'full']

    // Reset results
    const newResults: Record<PipelineVariant, VariantResult> = {
      text_only: { variant: 'text_only', status: 'running', progress: 0.1 },
      text_retrieval: { variant: 'text_retrieval', status: 'running', progress: 0.1 },
      text_image: { variant: 'text_image', status: 'running', progress: 0.1 },
      full: { variant: 'full', status: 'running', progress: 0.1 },
    }
    setResults(newResults)

    // Launch all 4 in parallel
    const newJobIds: Record<PipelineVariant, string | null> = {
      text_only: null,
      text_retrieval: null,
      text_image: null,
      full: null,
    }

    await Promise.all(
      variants.map(async (v) => {
        try {
          const job = await verifyClaim({
            text: claimText.trim(),
            options: {
              variant: v,
              enable_reasoning: true,
              store_result: false,
            },
          })
          newJobIds[v] = job.job_id
        } catch (err: unknown) {
          const e = err as { message?: string }
          setResults((prev) => ({
            ...prev,
            [v]: { ...prev[v], status: 'failed', error: e.message || 'Failed to start' },
          }))
        }
      })
    )

    setActiveJobIds(newJobIds)
  }

  const getVerdictDisplay = (report?: ReportEnvelope) => {
    const claim = report?.claims?.[0]
    if (!claim) return '-'
    const v = claim.verdict.toLowerCase()
    if (claim.abstained || v.includes('abstain')) {
      return (
        <span className="px-2 py-0.5 rounded text-xs font-semibold bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200">
          {t.verdicts.badgeAbstained}
        </span>
      )
    }
    if (v.includes('contra')) {
      return (
        <span className="px-2 py-0.5 rounded text-xs font-semibold bg-rose-100 text-rose-900 dark:bg-rose-950 dark:text-rose-200">
          {t.verdicts.badgeContradicted}
        </span>
      )
    }
    if (v.includes('verif') || v.includes('supp')) {
      return (
        <span className="px-2 py-0.5 rounded text-xs font-semibold bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-200">
          {t.verdicts.badgeVerified}
        </span>
      )
    }
    return (
      <span className="px-2 py-0.5 rounded text-xs font-semibold bg-slate-200 text-slate-800 dark:bg-slate-700 dark:text-slate-300">
        {t.verdicts.badgeInsufficient}
      </span>
    )
  }

  const getTotalDuration = (report?: ReportEnvelope) => {
    if (!report?.timings) return '-'
    const sum = Object.values(report.timings).reduce((acc, sec) => acc + sec, 0)
    return `${(sum * 1000).toFixed(0)} ms`
  }

  const variantsList: PipelineVariant[] = ['text_only', 'text_retrieval', 'text_image', 'full']

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
      {/* Title */}
      <div className="text-center mb-8">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
          {t.comparePage.title}
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          {t.comparePage.subtitle}
        </p>
      </div>

      {/* Input Form */}
      <form onSubmit={handleStartCompare} className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm mb-8 space-y-4">
        <label htmlFor="compare-input" className="text-sm font-semibold text-slate-900 dark:text-white block">
          {t.comparePage.inputLabel}
        </label>
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            id="compare-input"
            type="text"
            value={claimText}
            onChange={(e) => setClaimText(e.target.value)}
            placeholder="e.g., WHO declared drinking coffee prevents diabetes in 2026..."
            className="flex-1 text-sm rounded-xl border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-3 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            required
          />
          <button
            type="submit"
            disabled={isRunning || !claimText.trim()}
            className="px-6 py-3 rounded-xl text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 transition-colors shadow-md shadow-indigo-500/20 whitespace-nowrap"
          >
            {isRunning ? t.comparePage.runningVariants : t.comparePage.submitAll}
          </button>
        </div>
      </form>

      {/* Comparison Matrix Table */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden mb-8">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/60 font-semibold text-slate-700 dark:text-slate-300">
                <th className="p-4">{t.comparePage.colVariant}</th>
                <th className="p-4">{t.comparePage.colVerdict}</th>
                <th className="p-4">{t.comparePage.colConfidence}</th>
                <th className="p-4">{t.comparePage.colEvidence}</th>
                <th className="p-4">{t.comparePage.colTiming}</th>
                <th className="p-4">{t.comparePage.colDegraded}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {variantsList.map((v) => {
                const res = results[v]
                const isDone = res.status === 'done'
                const isFail = res.status === 'failed'
                const isProc = res.status === 'running'

                return (
                  <tr key={v} className="hover:bg-slate-50/50 dark:hover:bg-slate-800/30 transition-colors">
                    {/* Variant Name */}
                    <td className="p-4 font-semibold text-slate-900 dark:text-slate-100">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono">{v}</span>
                        {v === 'full' && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300 font-bold">
                            Default
                          </span>
                        )}
                      </div>
                    </td>

                    {/* Verdict */}
                    <td className="p-4">
                      {isDone ? (
                        getVerdictDisplay(res.report)
                      ) : isProc ? (
                        <span className="text-indigo-600 dark:text-indigo-400 font-medium animate-pulse">
                          Analyzing ({Math.round(res.progress * 100)}%)...
                        </span>
                      ) : isFail ? (
                        <span className="text-rose-600 font-medium">Failed</span>
                      ) : (
                        <span className="text-slate-400">Idle</span>
                      )}
                    </td>

                    {/* Confidence */}
                    <td className="p-4 font-mono font-medium">
                      {isDone && res.report?.claims?.[0] ? (
                        `${Math.round(res.report.claims[0].confidence * 100)}%`
                      ) : (
                        '-'
                      )}
                    </td>

                    {/* Sources Found */}
                    <td className="p-4 font-mono">
                      {isDone && res.report?.claims?.[0] ? (
                        `${(res.report.claims[0].conflict?.supporting?.length || 0) + (res.report.claims[0].conflict?.contradicting?.length || 0) || (res.report.claims[0].evidence?.length || 0)} sources`
                      ) : (
                        '-'
                      )}
                    </td>

                    {/* Duration */}
                    <td className="p-4 font-mono text-slate-600 dark:text-slate-400">
                      {isDone ? getTotalDuration(res.report) : '-'}
                    </td>

                    {/* Degraded Flags */}
                    <td className="p-4">
                      {isDone && res.report?.degraded && res.report.degraded.length > 0 ? (
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200">
                          {res.report.degraded.join(', ')}
                        </span>
                      ) : (
                        <span className="text-slate-400">None</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Epistemic Disclaimer */}
      <div className="rounded-2xl border border-indigo-200 dark:border-indigo-900 bg-indigo-50/40 dark:bg-indigo-950/20 p-6 space-y-2">
        <h3 className="text-sm font-bold text-indigo-950 dark:text-indigo-200 flex items-center space-x-2">
          <svg className="w-4 h-4 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>{t.comparePage.disclaimerTitle}</span>
        </h3>
        <p className="text-xs text-indigo-900/80 dark:text-indigo-300 leading-relaxed">
          {t.comparePage.disclaimerDesc}
        </p>
      </div>
    </div>
  )
}
