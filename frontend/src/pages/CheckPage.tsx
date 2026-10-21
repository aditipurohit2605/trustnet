import React, { useState, useEffect, useRef } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useCopy } from '../context/CopyContext'
import { verifyClaim, type PipelineVariant } from '../api/client'

interface RecentReportRef {
  id: string
  timestamp: number
  variant: string
}

export const CheckPage: React.FC = () => {
  const { t } = useCopy()
  const navigate = useNavigate()

  // Form State
  const [claimText, setClaimText] = useState('')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null)
  const [variant, setVariant] = useState<PipelineVariant>('full')
  const [languageHint, setLanguageHint] = useState<string>('')
  const [enableReasoning, setEnableReasoning] = useState(true)
  const [storeResult, setStoreResult] = useState(true)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Validation & Loading
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  // Drag and drop state
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Recent report references (strictly IDs and timestamps only, no claim text or images)
  const [recentChecks, setRecentChecks] = useState<RecentReportRef[]>(() => {
    try {
      const stored = localStorage.getItem('trustnet_recent_report_ids')
      return stored ? JSON.parse(stored) : []
    } catch {
      return []
    }
  })

  // Clean up object URL when image changes or unmounts
  useEffect(() => {
    return () => {
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl)
      }
    }
  }, [imagePreviewUrl])

  const handleImageSelect = (file: File) => {
    setErrorMsg(null)
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp']
    if (!validTypes.includes(file.type)) {
      setErrorMsg(t.checkPage.fileTypeError)
      return
    }

    const maxSize = 10 * 1024 * 1024 // 10MB
    if (file.size > maxSize) {
      setErrorMsg(t.checkPage.fileSizeError)
      return
    }

    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl)
    }

    setImageFile(file)
    setImagePreviewUrl(URL.createObjectURL(file))
  }

  const handleRemoveImage = () => {
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl)
    }
    setImageFile(null)
    setImagePreviewUrl(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleClearForm = () => {
    setClaimText('')
    handleRemoveImage()
    setErrorMsg(null)
    setVariant('full')
    setLanguageHint('')
    setEnableReasoning(true)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!claimText.trim()) {
      setErrorMsg(t.checkPage.emptyInputError)
      return
    }

    setIsSubmitting(true)
    setErrorMsg(null)

    try {
      const job = await verifyClaim({
        text: claimText.trim(),
        image: imageFile || undefined,
        options: {
          variant,
          language_hint: languageHint || undefined,
          enable_reasoning: enableReasoning,
          store_result: storeResult,
          store_image: false,
        },
      })

      // Navigate to polling page
      navigate(`/jobs/${job.job_id}`)
    } catch (err: unknown) {
      const e = err as { message?: string }
      setErrorMsg(e.message || t.errors.generic)
      setIsSubmitting(false)
    }
  }

  const handleClearRecent = () => {
    try {
      localStorage.removeItem('trustnet_recent_report_ids')
    } catch {
      // Ignore
    }
    setRecentChecks([])
  }

  const charCount = claimText.length
  const maxChars = 10000

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
      {/* Page Title */}
      <div className="text-center mb-8">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
          {t.checkPage.headerTitle}
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400 max-w-xl mx-auto">
          {t.checkPage.headerSubtitle}
        </p>
      </div>

      {/* Main Verification Form */}
      <form onSubmit={handleSubmit} className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm p-6 sm:p-8 space-y-6">
        {/* Error Alert */}
        {errorMsg && (
          <div role="alert" className="p-3.5 rounded-xl border border-rose-300 bg-rose-50 dark:border-rose-800 dark:bg-rose-950/40 text-xs text-rose-900 dark:text-rose-200 flex items-start space-x-2.5">
            <svg className="w-4 h-4 text-rose-600 dark:text-rose-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Claim Text Input */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label htmlFor="claim-text-input" className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              {t.checkPage.inputLabel} <span className="text-rose-500">*</span>
            </label>
            <span className={`text-xs font-mono ${charCount >= maxChars ? 'text-rose-600 font-bold' : 'text-slate-400'}`}>
              {charCount} / {maxChars} {t.checkPage.charCount}
            </span>
          </div>
          <textarea
            id="claim-text-input"
            value={claimText}
            onChange={(e) => setClaimText(e.target.value.slice(0, maxChars))}
            placeholder={t.checkPage.inputPlaceholder}
            rows={4}
            required
            className="w-full text-sm rounded-xl border border-slate-300 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-800/40 p-3.5 text-slate-900 dark:text-white placeholder-slate-400 focus:border-indigo-500 focus:bg-white dark:focus:bg-slate-900 focus:ring-2 focus:ring-indigo-500/20 transition-all outline-none"
          />
        </div>

        {/* Optional Image Dropzone */}
        <div className="space-y-2">
          <label className="text-sm font-semibold text-slate-900 dark:text-slate-100 block">
            {t.checkPage.imageUploadLabel}
          </label>

          {imagePreviewUrl ? (
            <div className="relative rounded-xl border border-slate-200 dark:border-slate-700 p-3 bg-slate-50 dark:bg-slate-800/50 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <img
                  src={imagePreviewUrl}
                  alt="Claim attachment preview"
                  className="w-16 h-16 object-cover rounded-lg border border-slate-200 dark:border-slate-700"
                />
                <div>
                  <p className="text-xs font-medium text-slate-900 dark:text-slate-100 truncate max-w-xs">
                    {imageFile?.name}
                  </p>
                  <p className="text-[11px] text-slate-500">
                    {imageFile ? (imageFile.size / (1024 * 1024)).toFixed(2) + ' MB' : ''}
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={handleRemoveImage}
                className="px-3 py-1.5 text-xs font-semibold text-rose-600 hover:text-rose-700 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-950/40 rounded-lg transition-colors"
              >
                {t.checkPage.removeImage}
              </button>
            </div>
          ) : (
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setIsDragOver(true)
              }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={(e) => {
                e.preventDefault()
                setIsDragOver(false)
                if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                  handleImageSelect(e.dataTransfer.files[0])
                }
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`rounded-xl border-2 border-dashed p-6 text-center cursor-pointer transition-colors ${
                isDragOver
                  ? 'border-indigo-500 bg-indigo-50/50 dark:bg-indigo-950/20'
                  : 'border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleImageSelect(e.target.files[0])
                  }
                }}
              />
              <svg className="w-8 h-8 mx-auto text-slate-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-xs font-medium text-slate-700 dark:text-slate-300">
                {isDragOver ? t.checkPage.dropzoneActive : t.checkPage.dropzoneIdle}
              </p>
              <p className="text-[11px] text-slate-400 mt-1">
                {t.checkPage.supportedFormats}
              </p>
            </div>
          )}
        </div>

        {/* Variant Selector */}
        <div className="space-y-3">
          <div>
            <label className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              {t.variants.title}
            </label>
            <p className="text-xs text-slate-500">{t.variants.subtitle}</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {(
              [
                { key: 'full', ...t.variants.full },
                { key: 'text_retrieval', ...t.variants.text_retrieval },
                { key: 'text_image', ...t.variants.text_image },
                { key: 'text_only', ...t.variants.text_only },
              ] as const
            ).map((item) => {
              const isSelected = variant === item.key
              return (
                <div
                  key={item.key}
                  onClick={() => setVariant(item.key as PipelineVariant)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? 'border-indigo-600 bg-indigo-50/50 dark:bg-indigo-950/30 dark:border-indigo-500 shadow-xs ring-1 ring-indigo-500'
                      : 'border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-bold text-slate-900 dark:text-slate-100">
                      {item.name}
                    </span>
                    <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
                      {item.badge}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
                    {item.description}
                  </p>
                </div>
              )
            })}
          </div>
        </div>

        {/* Collapsible Advanced Options */}
        <div className="border-t border-slate-200 dark:border-slate-800 pt-4">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center space-x-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400 hover:text-indigo-700"
          >
            <span>{t.checkPage.advancedOptionsTitle}</span>
            <svg
              className={`w-3.5 h-3.5 transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {showAdvanced && (
            <div className="mt-4 space-y-4 p-4 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-800">
              {/* Language Hint */}
              <div>
                <label className="text-xs font-semibold text-slate-800 dark:text-slate-200 block mb-1">
                  {t.checkPage.languageHintLabel}
                </label>
                <select
                  value={languageHint}
                  onChange={(e) => setLanguageHint(e.target.value)}
                  className="text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 p-2 text-slate-900 dark:text-white"
                >
                  <option value="">{t.checkPage.languageHintAuto}</option>
                  <option value="en">{t.checkPage.languageHintEn}</option>
                  <option value="hi">{t.checkPage.languageHintHi}</option>
                  <option value="hi-Latn">{t.checkPage.languageHintHinglish}</option>
                </select>
              </div>

              {/* Enable Reasoning Toggle */}
              <div className="flex items-start space-x-2.5">
                <input
                  type="checkbox"
                  id="enable-reasoning"
                  checked={enableReasoning}
                  onChange={(e) => setEnableReasoning(e.target.checked)}
                  className="mt-0.5 h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                />
                <div>
                  <label htmlFor="enable-reasoning" className="text-xs font-semibold text-slate-800 dark:text-slate-200 cursor-pointer">
                    {t.checkPage.enableReasoningLabel}
                  </label>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">
                    {t.checkPage.enableReasoningDesc}
                  </p>
                </div>
              </div>

              {/* Store Result Toggle + Privacy Notice */}
              <div className="flex items-start space-x-2.5 pt-2 border-t border-slate-200 dark:border-slate-700">
                <input
                  type="checkbox"
                  id="store-result"
                  checked={storeResult}
                  onChange={(e) => setStoreResult(e.target.checked)}
                  className="mt-0.5 h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                />
                <div>
                  <label htmlFor="store-result" className="text-xs font-semibold text-slate-800 dark:text-slate-200 cursor-pointer">
                    {t.checkPage.storeResultLabel}
                  </label>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400">
                    {t.checkPage.storeResultDesc}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-2">
          <button
            type="button"
            onClick={handleClearForm}
            className="px-4 py-2 text-xs font-medium text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
          >
            {t.checkPage.clearButton}
          </button>

          <button
            type="submit"
            disabled={isSubmitting || !claimText.trim()}
            className="px-6 py-2.5 rounded-xl text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-indigo-500/20 transition-all flex items-center space-x-2"
          >
            {isSubmitting && (
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
            )}
            <span>{isSubmitting ? t.checkPage.submitting : t.checkPage.submitButton}</span>
          </button>
        </div>
      </form>

      {/* Recent Checks Section (Opt-in references only) */}
      {recentChecks.length > 0 && (
        <div className="mt-8 pt-6 border-t border-slate-200 dark:border-slate-800">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                {t.checkPage.recentChecksTitle}
              </h3>
              <p className="text-[11px] text-slate-500">{t.checkPage.recentChecksNote}</p>
            </div>
            <button
              type="button"
              onClick={handleClearRecent}
              className="text-xs text-rose-600 hover:text-rose-700 font-medium"
            >
              {t.checkPage.clearRecentChecks}
            </button>
          </div>

          <div className="flex flex-wrap gap-2">
            {recentChecks.map((item) => (
              <Link
                key={item.id}
                to={`/reports/${item.id}`}
                className="px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-xs font-mono text-indigo-600 dark:text-indigo-400 hover:border-indigo-300 transition-colors"
              >
                Report #{item.id.slice(0, 8)} ({item.variant})
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
