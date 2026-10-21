import tseslint from 'typescript-eslint'

export default tseslint.config(
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{js,ts,jsx,tsx}'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'error',
      'no-restricted-properties': [
        'error',
        {
          property: 'dangerouslySetInnerHTML',
          message:
            'CRITICAL SECURITY MANDATE: dangerouslySetInnerHTML is strictly prohibited across TrustNet to prevent XSS vulnerabilities.',
        },
      ],
      'no-restricted-syntax': [
        'error',
        {
          selector: 'JSXAttribute[name.name="dangerouslySetInnerHTML"]',
          message:
            'CRITICAL SECURITY MANDATE: dangerouslySetInnerHTML is strictly prohibited across TrustNet to prevent XSS vulnerabilities.',
        },
      ],
    },
  },
  {
    ignores: ['dist/**', 'node_modules/**', 'src/api/types.ts'],
  }
)
