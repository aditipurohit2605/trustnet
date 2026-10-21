import React, { useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useCopy } from '../context/CopyContext'
import { useResearchMode } from '../context/ResearchModeContext'
import { getReport, deleteReport } from '../api/client'
import { ClaimCard } from '../components/ClaimCard'
import { DegradedBanner } from '../components/DegradedBanner'
import { ResearchPanel } from '../components/ResearchPanel'

export const ReportPage: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const { t } = useCopy()
  const { researchMode } = useResearchMode()
  const navigate = useNavigate()

  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [showModelVersions, setShowModelVersions] = useState(false)

  const { data: envelope, isLoading, isError, error } = useQuery({
    queryKey: ['report', id],
    queryFn: () => getReport(id!),
    enabled: !!id,
  })

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-20 text-center">
        <div className="animate-spin w-8 h-8 mx-auto text-indigo-600 mb-3 border-4 border-indigo-200 border-t-indigo-600 rounded-full" />
        <p className="text-sm text-slate-500">{t.common.loading}</p>
      </div>
    )
  }

  if (isError || !envelope) {
    return (
      <div className="max-w-xl mx-auto px-4 py-16 text-center">
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-8 space-y-4">
          <div className="w-12 h-12 rounded-full bg-rose-100 text-rose-600 mx-auto flex items-center justify-center">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <h2 className="text-lg font-bold text-slate-900 dark:text-white">
            {t.errors.reportNotFound}
          </h2>
          <p className="text-xs text-slate-500">
            {error?.message || t.errors.generic}
          </p>
          <Link
            to="/"
            className="inline-flex px-4 py-2 bg-indigo-600 text-white rounded-lg text-xs font-semibold"
          >
            {t.common.back}
          </Link>
        </div>
      </div>
    )
  }

  const claims = envelope.claims || []
  const totalClaims = claims.length

  // Calculate plain-language summary
  let unconfirmedCount = 0
  let contradictedCount = 0

  claims.forEach((c) => {
    const v = c.verdict.toLowerCase()
    if (c.abstained || v.includes('abstain') || v.includes('insuff')) {
      unconfirmedCount++
    } else if (v.includes('contra')) {
      contradictedCount++
    }
  })

  let summaryBannerText: string = t.reportPage.allSupported
  let summaryBannerStyle = 'bg-emerald-50 border-emerald-200 text-emerald-900 dark:bg-emerald-950/40 dark:border-emerald-800 dark:text-emerald-200'

  if (contradictedCount > 0) {
    summaryBannerText = t.reportPage.someContradicted
      .replace('{count}', String(contradictedCount))
      .replace('{total}', String(totalClaims))
    summaryBannerStyle = 'bg-rose-50 border-rose-200 text-rose-900 dark:bg-rose-950/40 dark:border-rose-800 dark:text-rose-200'
  } else if (unconfirmedCount === totalClaims && totalClaims > 0) {
    summaryBannerText = t.reportPage.allUnconfirmed
    summaryBannerStyle = 'bg-amber-50 border-amber-200 text-amber-900 dark:bg-amber-950/40 dark:border-amber-800 dark:text-amber-200'
  } else if (unconfirmedCount > 0) {
    summaryBannerText = t.reportPage.someUnconfirmed
      .replace('{count}', String(unconfirmedCount))
      .replace('{total}', String(totalClaims))
    summaryBannerStyle = 'bg-amber-50 border-amber-200 text-amber-900 dark:bg-amber-950/40 dark:border-amber-800 dark:text-amber-200'
  }

  const handlePrint = () => {
    window.print()
  }

  const handleDownloadJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(envelope, null, 2))
    const downloadAnchor = document.createElement('a')
    downloadAnchor.setAttribute('href', dataStr)
    downloadAnchor.setAttribute('download', `trustnet-report-${envelope.report_id}.json`)
    document.body.appendChild(downloadAnchor)
    downloadAnchor.click()
    downloadAnchor.remove()
  }

  const handleDelete = async () => {
    setIsDeleting(true)
    try {
      await deleteReport(envelope.report_id)
      setShowDeleteModal(false)
      navigate('/', { replace: true })
    } catch {
      alert('Failed to delete report from server.')
      setIsDeleting(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
      {/* Degraded Alert Banner if any component failed */}
      <DegradedBanner degraded={envelope.degraded} />

      {/* Top Plain-Language Summary Banner */}
      <div
        role="region"
        aria-label="Executive Summary"
        className={`mb-6 p-4 rounded-2xl border shadow-xs flex items-center justify-between gap-4 ${summaryBannerStyle}`}
      >
        <div className="flex items-center space-x-3">
          <svg className="w-6 h-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <h2 className="text-sm font-bold uppercase tracking-wider">{t.reportPage.summaryTitle}</h2>
            <p className="text-sm sm:text-base font-semibold mt-0.5">{summaryBannerText}</p>
          </div>
        </div>

        {/* Action Controls (Hidden in Print) */}
        <div className="flex items-center space-x-2 print:hidden flex-shrink-0">
          <button
            type="button"
            onClick={handlePrint}
            className="p-2 rounded-lg bg-white/80 dark:bg-slate-900/80 hover:bg-white text-slate-700 dark:text-slate-200 border border-slate-300 dark:border-slate-700 transition-colors"
            title={t.common.printReport}
            aria-label={t.common.printReport}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
            </svg>
          </button>

          <button
            type="button"
            onClick={handleDownloadJson}
            className="p-2 rounded-lg bg-white/80 dark:bg-slate-900/80 hover:bg-white text-slate-700 dark:text-slate-200 border border-slate-300 dark:border-slate-700 transition-colors"
            title={t.common.downloadJson}
            aria-label={t.common.downloadJson}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
          </button>

          <button
            type="button"
            onClick={() => setShowDeleteModal(true)}
            className="p-2 rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-600 border border-rose-200 transition-colors"
            title={t.common.delete}
            aria-label={t.common.delete}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>

      {/* Report Metadata Header */}
      <div className="mb-6 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 text-xs">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-3">
            <span className="font-semibold text-slate-900 dark:text-slate-100">
              {t.reportPage.reportTitle}:
            </span>
            <span className="font-mono text-slate-500">{envelope.report_id}</span>
            <span className="px-2 py-0.5 rounded font-mono font-semibold bg-indigo-50 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
              {envelope.variant}
            </span>
          </div>

          <div className="flex items-center space-x-4 text-slate-500">
            <span>
              {(envelope as unknown as { created_at?: string }).created_at
                ? new Date((envelope as unknown as { created_at: string }).created_at).toLocaleString()
                : new Date().toLocaleDateString()}
            </span>
            <button
              type="button"
              onClick={() => setShowModelVersions(!showModelVersions)}
              className="font-medium text-indigo-600 dark:text-indigo-400 hover:underline print:hidden"
            >
              Models ({Object.keys(envelope.model_versions || {}).length})
            </button>
          </div>
        </div>

        {/* Collapsible Model Versions */}
        {showModelVersions && envelope.model_versions && (
          <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-800 grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-[11px]">
            {Object.entries(envelope.model_versions).map(([k, v]) => (
              <div key={k} className="p-1.5 rounded bg-slate-50 dark:bg-slate-800">
                <span className="text-slate-400 block">{k}</span>
                <span className="text-slate-700 dark:text-slate-300 font-semibold">{String(v)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Claims List */}
      <div className="space-y-6">
        {claims.map((c, idx) => (
          <ClaimCard key={idx} reportId={envelope.report_id} claim={c} index={idx} />
        ))}
      </div>

      {/* Research / Inspector Mode Panel */}
      {researchMode && <ResearchPanel envelope={envelope} />}

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-xs">
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 max-w-md w-full p-6 shadow-xl space-y-4">
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              {t.reportPage.deleteModalTitle}
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {t.reportPage.deleteModalDesc}
            </p>
            <div className="flex items-center justify-end space-x-3 pt-3">
              <button
                type="button"
                onClick={() => setShowDeleteModal(false)}
                className="px-4 py-2 rounded-lg text-xs font-semibold text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
              >
                {t.common.cancel}
              </button>
              <button
                type="button"
                onClick={handleDelete}
                disabled={isDeleting}
                className="px-4 py-2 rounded-lg text-xs font-semibold text-white bg-rose-600 hover:bg-rose-700 disabled:opacity-50"
              >
                {isDeleting ? 'Deleting...' : t.reportPage.deleteConfirm}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
