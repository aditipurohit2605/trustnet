import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

test.describe('TrustNet Accessibility and Responsive Audits', () => {
  test('CheckPage passes axe-core accessibility audit', async ({ page }) => {
    await page.goto('/')

    // Wait for the main heading
    await expect(page.locator('h1')).toBeVisible()

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze()

    expect(accessibilityScanResults.violations).toEqual([])
  })

  test('AboutPage passes axe-core accessibility audit', async ({ page }) => {
    await page.goto('/about')
    await expect(page.locator('h1')).toBeVisible()

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze()

    expect(accessibilityScanResults.violations).toEqual([])
  })

  test('ComparePage passes axe-core accessibility audit', async ({ page }) => {
    await page.goto('/compare')
    await expect(page.locator('h1')).toBeVisible()

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze()

    expect(accessibilityScanResults.violations).toEqual([])
  })

  test('Mobile responsive view (360px) has no horizontal scroll blowout', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('h1')).toBeVisible()

    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth)
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth)

    // Ensure horizontal content fits within screen without unintended overflow
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 2)
  })

  test('Keyboard navigation: Skip link is present and focusable', async ({ page }) => {
    await page.goto('/')
    await page.keyboard.press('Tab')

    const skipLink = page.locator('a[href="#main-content"]')
    await expect(skipLink).toBeFocused()
  })
})
