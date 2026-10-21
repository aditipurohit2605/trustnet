import React, { createContext, useContext, useState, useEffect } from 'react'
import { enCopy, type CopyDictionary } from '../copy/en'
import { hiCopy } from '../copy/hi'

export type LanguageCode = 'en' | 'hi'

interface CopyContextType {
  lang: LanguageCode
  setLang: (lang: LanguageCode) => void
  copy: CopyDictionary
  t: CopyDictionary
}

const CopyContext = createContext<CopyContextType | null>(null)

export const CopyProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [lang, setLangState] = useState<LanguageCode>(() => {
    try {
      const saved = localStorage.getItem('trustnet_lang')
      if (saved === 'hi' || saved === 'en') return saved
    } catch {
      // In private browsing or storage disabled environments
    }
    return 'en'
  })

  const setLang = (newLang: LanguageCode) => {
    setLangState(newLang)
    try {
      localStorage.setItem('trustnet_lang', newLang)
    } catch {
      // Ignore
    }
  }

  useEffect(() => {
    document.documentElement.lang = lang
    if (lang === 'hi') {
      document.body.classList.add('font-devanagari')
    } else {
      document.body.classList.remove('font-devanagari')
    }
  }, [lang])

  const copy = lang === 'hi' ? hiCopy : enCopy

  return (
    <CopyContext.Provider value={{ lang, setLang, copy, t: copy }}>
      {children}
    </CopyContext.Provider>
  )
}

export const useCopy = (): CopyContextType => {
  const context = useContext(CopyContext)
  if (!context) {
    throw new Error('useCopy must be used within a CopyProvider')
  }
  return context
}
