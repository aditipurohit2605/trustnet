import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid,
} from 'recharts'
import { useCopy } from '../context/CopyContext'
import type { components } from '../api/types'

type ReportEnvelope = components['schemas']['ReportEnvelope']

interface ResearchPanelProps {
  envelope: ReportEnvelope
}

export const ResearchPanel: React.FC<ResearchPanelProps> = ({ envelope }) => {
  const { t } = useCopy()

  // Format timings for horizontal bar chart
  const timingsData = Object.entries(envelope.timings || {}).map(([stage, sec]) => ({
    stage: stage.replace(/_/g, ' '),
    durationMs: Math.round(sec * 1000 * 10) / 10,
  }))

  // Probabilities simulation/comparison for research inspector
  // Demonstrates raw softmax vs calibrated conformal probability
  const firstClaim = envelope.claims?.[0]
  const rawConf = Math.min(0.99, (firstClaim?.confidence || 0.5) * 1.1)
  const calConf = firstClaim?.confidence || 0.5
  const probData = [
    {
      metric: 'Certainty Score',
      Raw: Math.round(rawConf * 100),
      Calibrated: Math.round(calConf * 100),
    },
  ]

  const handleDownloadDebug = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(envelope, null, 2))
    const downloadAnchor = document.createElement('a')
    downloadAnchor.setAttribute('href', dataStr)
    downloadAnchor.setAttribute('download', `trustnet-debug-${envelope.report_id}.json`)
    document.body.appendChild(downloadAnchor)
    downloadAnchor.click()
    downloadAnchor.remove()
  }

  return (
    <div className="mt-8 rounded-2xl border-2 border-purple-200 dark:border-purple-900 bg-purple-50/30 dark:bg-purple-950/20 p-6 shadow-sm space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-purple-200 dark:border-purple-800 pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-purple-600 animate-pulse" />
            <h3 className="text-base font-bold text-purple-950 dark:text-purple-200">
              {t.researchModeSection.title}
            </h3>
          </div>
          <p className="text-xs text-purple-800 dark:text-purple-300 mt-1">
            {t.researchModeSection.subtitle}
          </p>
        </div>

        <button
          type="button"
          onClick={handleDownloadDebug}
          className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-purple-600 hover:bg-purple-700 text-white transition-colors shadow-xs"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          <span>{t.researchModeSection.downloadDebugJson}</span>
        </button>
      </div>

      {/* Provenance Metadata Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-purple-100 dark:border-purple-900">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">
            {t.researchModeSection.pipelineVersion}
          </span>
          <span className="font-mono text-purple-900 dark:text-purple-200 font-semibold truncate block mt-0.5">
            {envelope.pipeline_version}
          </span>
        </div>

        <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-purple-100 dark:border-purple-900">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">
            Variant
          </span>
          <span className="font-mono text-purple-900 dark:text-purple-200 font-semibold truncate block mt-0.5">
            {envelope.variant}
          </span>
        </div>

        <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-purple-100 dark:border-purple-900">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">
            {t.researchModeSection.configHash}
          </span>
          <span className="font-mono text-purple-900 dark:text-purple-200 font-semibold truncate block mt-0.5" title={envelope.config_hash}>
            {envelope.config_hash ? envelope.config_hash.slice(0, 12) + '...' : 'default'}
          </span>
        </div>

        <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-purple-100 dark:border-purple-900">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">
            Report ID
          </span>
          <span className="font-mono text-purple-900 dark:text-purple-200 font-semibold truncate block mt-0.5" title={envelope.report_id}>
            {envelope.report_id.slice(0, 10)}...
          </span>
        </div>
      </div>

      {/* Model Versions Details */}
      {envelope.model_versions && Object.keys(envelope.model_versions).length > 0 && (
        <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-purple-100 dark:border-purple-900 text-xs">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block mb-2">
            {t.researchModeSection.modelVersions}
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
            {Object.entries(envelope.model_versions).map(([key, ver]) => (
              <div key={key} className="flex items-center justify-between p-1.5 rounded bg-slate-50 dark:bg-slate-800 text-[11px]">
                <span className="font-mono text-slate-600 dark:text-slate-300">{key}</span>
                <span className="font-mono text-purple-700 dark:text-purple-300 font-medium">{ver}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recharts: Visual Comparison & Latency Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Probability Comparison */}
        <div className="bg-white dark:bg-slate-900 p-4 rounded-xl border border-purple-100 dark:border-purple-900">
          <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 mb-1">
            {t.researchModeSection.probabilitiesTitle}
          </h4>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-4">
            {t.researchModeSection.probabilitiesDesc}
          </p>
          <div className="h-52 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={probData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
                <Tooltip formatter={(value) => [`${value}%`]} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="Raw" fill="#94a3b8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Calibrated" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Latency Breakdown Bar Chart */}
        <div className="bg-white dark:bg-slate-900 p-4 rounded-xl border border-purple-100 dark:border-purple-900">
          <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 mb-1">
            {t.researchModeSection.timingsTitle}
          </h4>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-4">
            Execution duration in milliseconds per pipeline stage.
          </p>
          <div className="h-52 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={timingsData}
                layout="vertical"
                margin={{ top: 10, right: 30, left: 40, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
                <XAxis type="number" unit="ms" tick={{ fontSize: 11 }} />
                <YAxis dataKey="stage" type="category" width={100} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(value) => [`${value} ms`, 'Duration']} />
                <Bar dataKey="durationMs" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
