import React, { useState } from 'react'
import { useCopy } from '../context/CopyContext'
import { ConfidenceGauge } from './ConfidenceGauge'
import { EvidenceSection } from './EvidenceSection'
import { GapsChecklist } from './GapsChecklist'
import { TemporalInfoSection } from './TemporalInfoSection'
import { WhatIfSection } from './WhatIfSection'
import { StyleSignalsSection } from './StyleSignalsSection'
import { submitFeedback } from '../api/client'
import type { components } from '../api/types'

type ClaimReport = components['schemas']['ClaimReport']

interface ClaimCardProps {
  reportId: string
  claim: ClaimReport
  index: number
}

export const ClaimCard: React.FC<ClaimCardProps> = ({ reportId, claim, index }) => {
  const { t } = useCopy()
  const [isExpanded, setIsExpanded] = useState(true)

  // Feedback form state
  const [feedbackRating, setFeedbackRating] = useState<'thumbs_up' | 'thumbs_down' | null>(null)
  const [feedbackComment, setFeedbackComment] = useState('')
  const [feedbackConsent, setFeedbackConsent] = useState(false)
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false)
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false)
  const [feedbackError, setFeedbackError] = useState<string | null>(null)

  const isAbstained = claim.abstained || claim.verdict.toLowerCase().includes('abstain')
  const verdictLower = claim.verdict.toLowerCase()

  // Verbatim wording lookup
  let verdictCopy = t.verdicts.insufficient
  let badgeLabel = t.verdicts.badgeInsufficient
  let badgeColor = 'bg-slate-100 text-slate-800 border-slate-300 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-700'

  if (isAbstained) {
    verdictCopy = t.verdicts.abstained
    badgeLabel = t.verdicts.badgeAbstained
    badgeColor = 'bg-amber-100 text-amber-900 border-amber-300 dark:bg-amber-950 dark:text-amber-200 dark:border-amber-700'
  } else if (verdictLower.includes('contra')) {
    verdictCopy = t.verdicts.contradicted
    badgeLabel = t.verdicts.badgeContradicted
    badgeColor = 'bg-rose-100 text-rose-900 border-rose-300 dark:bg-rose-950 dark:text-rose-200 dark:border-rose-700'
  } else if (verdictLower.includes('verif') || verdictLower.includes('supp')) {
    verdictCopy = t.verdicts.verified
    badgeLabel = t.verdicts.badgeVerified
    badgeColor = 'bg-emerald-100 text-emerald-900 border-emerald-300 dark:bg-emerald-950 dark:text-emerald-200 dark:border-emerald-700'
  }

  // Share advice styling
  const shareAdvice = (claim.share_advice || 'share_with_caution').toLowerCase()
  let shareBadge = t.shareAdvice.share_with_caution
  let shareDesc = t.shareAdvice.cautionDescription
  let shareCardStyle = 'bg-amber-50/80 border-amber-200 text-amber-900 dark:bg-amber-950/40 dark:border-amber-800 dark:text-amber-200'

  if (shareAdvice.includes('safe')) {
    shareBadge = t.shareAdvice.safe_to_share
    shareDesc = t.shareAdvice.safeDescription
    shareCardStyle = 'bg-emerald-50/80 border-emerald-200 text-emerald-900 dark:bg-emerald-950/40 dark:border-emerald-800 dark:text-emerald-200'
  } else if (shareAdvice.includes('do_not') || shareAdvice.includes('donot')) {
    shareBadge = t.shareAdvice.do_not
    shareDesc = t.shareAdvice.doNotDescription
    shareCardStyle = 'bg-rose-50/80 border-rose-200 text-rose-900 dark:bg-rose-950/40 dark:border-rose-800 dark:text-rose-200'
  }

  // Language display
  const langCode = claim.claim?.language_info?.code || claim.language_info?.code
  let langLabel = 'English'
  if (langCode === 'hi') langLabel = 'हिन्दी (Hindi)'
  else if (langCode === 'hi-Latn' || claim.claim?.language_info?.is_code_mixed || claim.language_info?.is_code_mixed) langLabel = 'Hinglish / Mixed'

  const claimStatement = claim.claim?.text || ''
  const origText = claim.claim?.original_text || claim.original_text
  const hasOriginal = origText && origText !== claimStatement

  const handleFeedbackSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!feedbackRating) return

    setFeedbackSubmitting(true)
    setFeedbackError(null)

    try {
      await submitFeedback(reportId, {
        claim_id: claim.claim?.id,
        comment: feedbackComment,
        consent_to_store: feedbackConsent,
        suggested_verdict: feedbackRating === 'thumbs_up' ? String(claim.verdict) : undefined,
      })
      setFeedbackSubmitted(true)
    } catch {
      setFeedbackError('Unable to submit feedback. Please try again.')
    } finally {
      setFeedbackSubmitting(false)
    }
  }

  return (
    <article
      className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden transition-all duration-200 print:border print:shadow-none print:break-inside-avoid"
      aria-labelledby={`claim-heading-${index}`}
    >
      {/* Top Header Bar */}
      <div className="p-5 sm:p-6 border-b border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-800/20">
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
          <div className="space-y-2 flex-1">
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300">
                {t.reportPage.claimNumber.replace('{number}', String(index + 1))}
              </span>
              <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                {langLabel}
              </span>
            </div>

            {/* Claim Statement */}
            <h3
              id={`claim-heading-${index}`}
              className="text-lg font-semibold text-slate-900 dark:text-white leading-snug"
            >
              "{claimStatement}"
            </h3>

            {/* Translated or original text if different */}
            {hasOriginal && (
              <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                <span className="font-semibold text-slate-600 dark:text-slate-300 mr-1">
                  {t.reportPage.originalTextLabel}:
                </span>
                <span className="italic">"{origText}"</span>
              </div>
            )}
          </div>

          {/* Model Confidence Meter */}
          <div className="sm:text-right flex-shrink-0">
            <ConfidenceGauge confidence={claim.confidence} />
          </div>
        </div>

        {/* Verdict Banner */}
        <div className="mt-4 pt-4 border-t border-slate-200/60 dark:border-slate-800/60 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center space-x-2.5">
            <span
              className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold border uppercase tracking-wider ${badgeColor}`}
            >
              {badgeLabel}
            </span>
            <span className="text-xs sm:text-sm font-semibold text-slate-800 dark:text-slate-200">
              {verdictCopy}
            </span>
          </div>

          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300 inline-flex items-center space-x-1 focus:outline-none print:hidden"
            aria-expanded={isExpanded}
          >
            <span>{isExpanded ? 'Collapse Details' : 'View Full Reasoning'}</span>
            <svg
              className={`w-3.5 h-3.5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>

      {/* Special Layout for Abstained Claims */}
      {isAbstained && (
        <div className="mx-5 sm:mx-6 mt-4 p-4 rounded-xl border border-amber-300 bg-amber-50/70 dark:border-amber-700/60 dark:bg-amber-950/30">
          <div className="flex items-start space-x-3">
            <svg
              className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-amber-900 dark:text-amber-200">
                {t.reportPage.abstainedAlertTitle}
              </h4>
              <p className="mt-1 text-xs text-amber-800 dark:text-amber-300 leading-relaxed">
                {t.reportPage.abstainedAlertDesc}
              </p>
              {claim.abstention_reason && (
                <div className="mt-2 text-xs font-mono font-medium text-amber-900 dark:text-amber-200 bg-amber-100/70 dark:bg-amber-900/40 px-2.5 py-1 rounded">
                  Reason: {claim.abstention_reason}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Share Advice Guidance Banner */}
      <div className="mx-5 sm:mx-6 mt-4">
        <div className={`p-3.5 rounded-xl border flex items-start space-x-3 ${shareCardStyle}`}>
          <div className="pt-0.5">
            {shareAdvice.includes('safe') ? (
              <svg className="w-5 h-5 text-emerald-600 dark:text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            ) : shareAdvice.includes('do_not') || shareAdvice.includes('donot') ? (
              <svg className="w-5 h-5 text-rose-600 dark:text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
              </svg>
            ) : (
              <svg className="w-5 h-5 text-amber-600 dark:text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            )}
          </div>
          <div className="flex-1 text-xs">
            <span className="font-bold text-sm block mb-0.5">{shareBadge}</span>
            <p className="opacity-90">{shareDesc}</p>
          </div>
        </div>
      </div>

      {/* Accordion / Reasoning Section (Always visible in Print, toggleable in Screen) */}
      <div className={`p-5 sm:p-6 space-y-6 ${isExpanded ? 'block' : 'hidden print:block'}`}>
        {/* Evidence Section */}
        <EvidenceSection
          supporting={claim.conflict?.supporting || claim.evidence || []}
          contradicting={claim.conflict?.contradicting || []}
          conflict={claim.conflict}
          sourceWeights={claim.source_weights || []}
        />

        {/* Gaps Checklist */}
        <GapsChecklist gaps={claim.gaps || []} />

        {/* Temporal Info */}
        <TemporalInfoSection temporal={claim.temporal} />

        {/* Counterfactuals */}
        <WhatIfSection counterfactuals={claim.counterfactuals || []} />

        {/* Style Signals */}
        <StyleSignalsSection signals={claim.manipulation || []} />

        {/* Inline Feedback Form */}
        <div className="mt-8 pt-6 border-t border-slate-200 dark:border-slate-800 print:hidden">
          <div className="bg-slate-50 dark:bg-slate-800/40 p-4 rounded-xl border border-slate-200/80 dark:border-slate-700/80">
            <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 mb-1">
              {t.reportPage.feedbackTitle}
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
              {t.reportPage.feedbackSubtitle}
            </p>

            {feedbackSubmitted ? (
              <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-200 text-xs font-medium">
                {t.reportPage.feedbackSuccess}
              </div>
            ) : (
              <form onSubmit={handleFeedbackSubmit} className="space-y-3">
                <div className="flex items-center space-x-2">
                  <button
                    type="button"
                    onClick={() => setFeedbackRating('thumbs_up')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold border flex items-center space-x-1.5 transition-colors ${
                      feedbackRating === 'thumbs_up'
                        ? 'bg-emerald-100 text-emerald-900 border-emerald-400 dark:bg-emerald-950 dark:text-emerald-200'
                        : 'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    <span>👍</span>
                    <span>{t.reportPage.feedbackThumbsUp}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setFeedbackRating('thumbs_down')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold border flex items-center space-x-1.5 transition-colors ${
                      feedbackRating === 'thumbs_down'
                        ? 'bg-rose-100 text-rose-900 border-rose-400 dark:bg-rose-950 dark:text-rose-200'
                        : 'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    <span>👎</span>
                    <span>{t.reportPage.feedbackThumbsDown}</span>
                  </button>
                </div>

                <div>
                  <textarea
                    value={feedbackComment}
                    onChange={(e) => setFeedbackComment(e.target.value)}
                    placeholder={t.reportPage.feedbackCommentLabel}
                    rows={2}
                    className="w-full text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 p-2.5 text-slate-900 dark:text-white placeholder-slate-400 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  />
                </div>

                <div className="flex items-start space-x-2">
                  <input
                    type="checkbox"
                    id={`feedback-consent-${index}`}
                    checked={feedbackConsent}
                    onChange={(e) => setFeedbackConsent(e.target.checked)}
                    className="mt-0.5 h-3.5 w-3.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  <label
                    htmlFor={`feedback-consent-${index}`}
                    className="text-[11px] text-slate-600 dark:text-slate-400 cursor-pointer"
                  >
                    {t.reportPage.feedbackConsentLabel}{' '}
                    <span className="text-slate-400">({t.reportPage.feedbackConsentNote})</span>
                  </label>
                </div>

                {feedbackError && (
                  <p className="text-xs text-rose-600 dark:text-rose-400">{feedbackError}</p>
                )}

                <button
                  type="submit"
                  disabled={!feedbackRating || feedbackSubmitting}
                  className="px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white rounded-lg text-xs font-medium transition-colors"
                >
                  {feedbackSubmitting ? 'Submitting...' : t.common.submit}
                </button>
              </form>
            )}
          </div>
        </div>
      </div>
    </article>
  )
}
