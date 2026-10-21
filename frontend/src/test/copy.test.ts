import { describe, it, expect } from 'vitest'
import { enCopy } from '../copy/en'
import { hiCopy } from '../copy/hi'

/**
 * Recursively extracts all keys from an object to ensure exact parity between locales.
 */
function getDeepKeys(obj: Record<string, unknown>, prefix = ''): string[] {
  return Object.keys(obj).flatMap((key) => {
    const fullPath = prefix ? `${prefix}.${key}` : key
    const val = obj[key]
    if (val && typeof val === 'object' && !Array.isArray(val)) {
      return getDeepKeys(val as Record<string, unknown>, fullPath)
    }
    return [fullPath]
  })
}

describe('Copy Subsystem and Epistemic Integrity', () => {
  it('has exact key parity between English and Hindi dictionaries', () => {
    const enKeys = getDeepKeys(enCopy as unknown as Record<string, unknown>).sort()
    const hiKeys = getDeepKeys(hiCopy as unknown as Record<string, unknown>).sort()

    expect(hiKeys).toEqual(enKeys)
  })

  it('strictly adheres to verbatim verdict texts in English', () => {
    expect(enCopy.verdicts.contradicted).toBe('Reliable sources we searched disagree with this claim')
    expect(enCopy.verdicts.insufficient).toBe("We couldn't confirm this. Don't share it as confirmed")
    expect(enCopy.verdicts.abstained).toBe("We couldn't confirm this. Don't share it as confirmed")
    expect(enCopy.verdicts.verified).toBe('Supported by the sources we searched')
  })

  it('strictly adheres to verbatim verdict texts in Hindi', () => {
    expect(hiCopy.verdicts.contradicted).toBe('हमारे द्वारा खोजे गए विश्वसनीय स्रोत इस दावे से असहमत हैं')
    expect(hiCopy.verdicts.insufficient).toBe('हम इसकी पुष्टि नहीं कर सके। इसे पुष्ट मानकर साझा न करें')
    expect(hiCopy.verdicts.abstained).toBe('हम इसकी पुष्टि नहीं कर सके। इसे पुष्ट मानकर साझा न करें')
    expect(hiCopy.verdicts.verified).toBe('हमारे द्वारा खोजे गए स्रोतों द्वारा समर्थित')
  })

  it('strictly bans "true", "false", and "fake" from verdict copy', () => {
    const forbiddenWords = ['true', 'false', 'fake', 'सच', 'झूठ', 'फेक']
    
    // Check verdicts object in English and Hindi
    const verdictValuesEn = Object.values(enCopy.verdicts)
    const verdictValuesHi = Object.values(hiCopy.verdicts)

    for (const val of [...verdictValuesEn, ...verdictValuesHi]) {
      const lower = val.toLowerCase()
      for (const forbidden of forbiddenWords) {
        // Assert the forbidden word does not appear as a standalone verdict or adjective
        const regex = new RegExp(`\\b${forbidden}\\b`, 'i')
        expect(regex.test(lower)).toBe(false)
      }
    }
  })

  it('clearly explains model confidence as certainty given evidence, not absolute truth', () => {
    expect(enCopy.confidence.label).toBe('Model confidence')
    expect(enCopy.confidence.tooltip).toContain('NOT the factual truth of the claim')
    expect(enCopy.aboutPage.confidenceText).toContain('NOT the probability that a claim is true in reality')
  })

  it('clearly contains style signals disclaimer that style is not evidence about truth', () => {
    expect(enCopy.reportPage.styleSignalsDisclaimer).toBe(
      'Style signals are NOT evidence about truth; they only describe how the claim is written.'
    )
    expect(hiCopy.reportPage.styleSignalsDisclaimer).toBe(
      'शैली संकेत सत्य के प्रमाण नहीं हैं; वे केवल यह दर्शाते हैं कि दावा किस शैली में लिखा गया है।'
    )
  })
})
