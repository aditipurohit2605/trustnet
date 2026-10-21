import React, { useState, useEffect } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useCopy } from '../context/CopyContext'
import { getJob } from '../api/client'

export const ProgressPage: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const { t } = useCopy()
  const navigate = useNavigate()

  const [elapsedSeconds, setElapsedSeconds] = useState(0)

  // Track elapsed time
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1)
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  // Poll job status every 1000ms while running or queued
  const { data: job, error, isError, refetch } = useQuery({
    queryKey: ['job', id],
    queryFn: () => getJob(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (status === 'done' || status === 'failed') {
        return false
      }
      return 1000
    },
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
  })

  // When job completes, record reference and redirect
  useEffect(() => {
    if (job?.status === 'done' && job.report) {
      try {
        const existing = JSON.parse(localStorage.getItem('trustnet_recent_report_ids') || '[]')
        const updated = [
          {
            id: job.report.report_id,
            timestamp: Date.now(),
            variant: job.report.variant,
          },
          ...existing.filter((item: { id: string }) => item.id !== job.report?.report_id),
        ].slice(0, 10)
        localStorage.setItem('trustnet_recent_report_ids', JSON.stringify(updated))
      } catch {
        // Ignore storage errors
      }

      navigate(`/reports/${job.report.report_id}`, { replace: true })
    }
  }, [job, navigate])

  const status = job?.status || 'queued'
  const progressPct = Math.round((job?.progress || 0) * 100)
  const isQueued = status === 'queued'

  return (
    <div className="max-w-2xl mx-auto px-4 py-16 text-center">
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm p-8 space-y-6">
        {/* Animated Status Icon */}
        <div className="mx-auto flex items-center justify-center">
          {isQueued ? (
            <div className="w-16 h-16 rounded-2xl bg-amber-100 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400 flex items-center justify-center shadow-md animate-pulse">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          ) : status === 'failed' ? (
            <div className="w-16 h-16 rounded-2xl bg-rose-100 dark:bg-rose-950/60 text-rose-600 dark:text-rose-400 flex items-center justify-center shadow-md">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
          ) : (
            <div className="w-16 h-16 rounded-2xl bg-indigo-100 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center shadow-md">
              <svg className="w-8 h-8 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
            </div>
          )}
        </div>

        {/* Status Headings */}
        <div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-white">
            {isQueued
              ? t.progressPage.queuedTitle
              : status === 'failed'
              ? t.progressPage.failedTitle
              : t.progressPage.runningTitle}
          </h2>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            {isQueued
              ? t.progressPage.queuedDesc
              : status === 'failed'
              ? job?.error || t.progressPage.failedDesc
              : t.progressPage.runningDesc}
          </p>
        </div>

        {/* Progress Bar & Stage indicator */}
        {status !== 'failed' && (
          <div className="space-y-3 pt-2">
            <div className="flex items-center justify-between text-xs font-medium text-slate-600 dark:text-slate-400">
              <span>
                {t.progressPage.stageLabel}:{' '}
                <strong className="text-indigo-600 dark:text-indigo-400 capitalize">
                  {job?.stage ? job.stage.replace(/_/g, ' ') : isQueued ? 'Waiting in Queue' : 'Initializing'}
                </strong>
              </span>
              <span>{progressPct}%</span>
            </div>

            <div
              className="w-full bg-slate-200 dark:bg-slate-700 h-2.5 rounded-full overflow-hidden"
              role="progressbar"
              aria-valuenow={progressPct}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`${t.progressPage.progressLabel}: ${progressPct}%`}
            >
              <div
                className="bg-indigo-600 h-full rounded-full transition-all duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        )}

        {/* Elapsed Timer & Job Identifier */}
        <div className="flex items-center justify-between text-xs text-slate-400 pt-4 border-t border-slate-100 dark:border-slate-800">
          <span>
            {t.progressPage.timeElapsed}: <strong>{elapsedSeconds}s</strong>
          </span>
          <span className="font-mono text-[11px]">Job: {id?.slice(0, 8)}...</span>
        </div>

        {/* Error / Backoff Recovery */}
        {isError && (
          <div className="p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-lg text-xs text-amber-900 dark:text-amber-200 flex items-center justify-between">
            <span>{error?.message || t.progressPage.networkError}</span>
            <button
              type="button"
              onClick={() => refetch()}
              className="px-2.5 py-1 rounded bg-amber-200 dark:bg-amber-800 font-semibold"
            >
              {t.progressPage.retryButton}
            </button>
          </div>
        )}

        {/* Cancel / Return */}
        <div className="pt-2">
          <Link
            to="/"
            className="text-xs font-semibold text-slate-500 hover:text-slate-800 dark:hover:text-white"
          >
            {t.progressPage.cancelButton}
          </Link>
        </div>
      </div>
    </div>
  )
}
