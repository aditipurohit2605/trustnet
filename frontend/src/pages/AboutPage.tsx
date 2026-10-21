import React from 'react'
import { useCopy } from '../context/CopyContext'

export const AboutPage: React.FC = () => {
  const { t } = useCopy()

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-12 space-y-12">
      {/* Title */}
      <div className="text-center">
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
          {t.aboutPage.title}
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400 max-w-xl mx-auto">
          {t.aboutPage.subtitle}
        </p>
      </div>

      {/* What TrustNet Is & Is Not */}
      <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center space-x-2">
          <span className="w-2 h-5 rounded-full bg-indigo-600" />
          <span>{t.aboutPage.whatIsTitle}</span>
        </h2>
        <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          {t.aboutPage.whatIsText}
        </p>
      </section>

      {/* Corpus Coverage & Boundaries */}
      <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center space-x-2">
          <span className="w-2 h-5 rounded-full bg-sky-600" />
          <span>{t.aboutPage.corpusTitle}</span>
        </h2>
        <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          {t.aboutPage.corpusText}
        </p>
      </section>

      {/* Interpreting Model Confidence vs Truth */}
      <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center space-x-2">
          <span className="w-2 h-5 rounded-full bg-purple-600" />
          <span>{t.aboutPage.confidenceTitle}</span>
        </h2>
        <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          {t.aboutPage.confidenceText}
        </p>
      </section>

      {/* Four Standard Outcomes */}
      <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center space-x-2">
          <span className="w-2 h-5 rounded-full bg-emerald-600" />
          <span>{t.aboutPage.outcomesTitle}</span>
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/50 dark:border-emerald-800 dark:bg-emerald-950/20 text-xs">
            <span className="font-bold text-emerald-900 dark:text-emerald-200 block mb-1">
              {t.verdicts.badgeVerified}
            </span>
            <p className="text-emerald-800 dark:text-emerald-300">
              {t.aboutPage.outcomeVerified}
            </p>
          </div>

          <div className="p-4 rounded-xl border border-rose-200 bg-rose-50/50 dark:border-rose-800 dark:bg-rose-950/20 text-xs">
            <span className="font-bold text-rose-900 dark:text-rose-200 block mb-1">
              {t.verdicts.badgeContradicted}
            </span>
            <p className="text-rose-800 dark:text-rose-300">
              {t.aboutPage.outcomeContradicted}
            </p>
          </div>

          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-800/40 text-xs">
            <span className="font-bold text-slate-900 dark:text-slate-200 block mb-1">
              {t.verdicts.badgeInsufficient}
            </span>
            <p className="text-slate-700 dark:text-slate-300">
              {t.aboutPage.outcomeInsufficient}
            </p>
          </div>

          <div className="p-4 rounded-xl border border-amber-200 bg-amber-50/50 dark:border-amber-800 dark:bg-amber-950/20 text-xs">
            <span className="font-bold text-amber-900 dark:text-amber-200 block mb-1">
              {t.verdicts.badgeAbstained}
            </span>
            <p className="text-amber-800 dark:text-amber-300">
              {t.aboutPage.outcomeAbstained}
            </p>
          </div>
        </div>
      </section>

      {/* Known Limitations */}
      <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center space-x-2">
          <span className="w-2 h-5 rounded-full bg-amber-500" />
          <span>{t.aboutPage.limitationsTitle}</span>
        </h2>
        <ul className="space-y-2.5 text-xs text-slate-600 dark:text-slate-300">
          {t.aboutPage.limitationsList.map((item, idx) => (
            <li key={idx} className="flex items-start space-x-2">
              <span className="text-indigo-600 font-bold">•</span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </section>

      {/* Privacy Architecture */}
      <section className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center space-x-2">
          <span className="w-2 h-5 rounded-full bg-teal-600" />
          <span>{t.aboutPage.privacyTitle}</span>
        </h2>
        <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          {t.aboutPage.privacyText}
        </p>
      </section>
    </div>
  )
}
