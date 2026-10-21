import React, { createContext, useContext, useState } from 'react'

interface ResearchModeContextType {
  researchMode: boolean
  setResearchMode: (value: boolean | ((prev: boolean) => boolean)) => void
  toggleResearchMode: () => void
}

const ResearchModeContext = createContext<ResearchModeContextType | null>(null)

export const ResearchModeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [researchMode, setResearchMode] = useState<boolean>(() => {
    try {
      return localStorage.getItem('trustnet_research_mode') === 'true'
    } catch {
      return false
    }
  })

  const updateResearchMode = (value: boolean | ((prev: boolean) => boolean)) => {
    setResearchMode((prev) => {
      const next = typeof value === 'function' ? value(prev) : value
      try {
        localStorage.setItem('trustnet_research_mode', String(next))
      } catch {
        // Ignore storage errors
      }
      return next
    })
  }

  const toggleResearchMode = () => {
    updateResearchMode((prev) => !prev)
  }

  return (
    <ResearchModeContext.Provider
      value={{
        researchMode,
        setResearchMode: updateResearchMode,
        toggleResearchMode,
      }}
    >
      {children}
    </ResearchModeContext.Provider>
  )
}

export const useResearchMode = (): ResearchModeContextType => {
  const context = useContext(ResearchModeContext)
  if (!context) {
    throw new Error('useResearchMode must be used within a ResearchModeProvider')
  }
  return context
}
