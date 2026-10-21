/**
 * Typed API Client for TrustNet Stage 11 Frontend.
 *
 * All request/response types derive strictly from generated OpenAPI schemas in `types.ts`.
 * Explicitly parses and maps HTTP error status codes (401, 413, 422, 429, 5xx) and network
 * disconnects to distinct, strongly-typed error classes.
 */

import type { components, paths } from './types'

// Re-export core schema types for ergonomic consumption across the app
export type JobResponse = components['schemas']['JobResponse']
export type ReportEnvelope = components['schemas']['ReportEnvelope']
export type ClaimReport = components['schemas']['ClaimReport']
export type Claim = components['schemas']['Claim']
export type Evidence = components['schemas']['Evidence']
export type Gap = components['schemas']['Gap']
export type ConflictInfo = components['schemas']['ConflictInfo']
export type TemporalInfo = components['schemas']['TemporalInfo']
export type ManipulationTag = components['schemas']['ManipulationTag']
export type Counterfactual = components['schemas']['Counterfactual']
export type SourceWeight = components['schemas']['SourceWeight']
export type LanguageInfo = components['schemas']['LanguageInfo']
export type NormalizedText = components['schemas']['NormalizedText']
export type FeedbackIn = components['schemas']['FeedbackIn']
export type PipelineVariant = 'text_only' | 'text_retrieval' | 'text_image' | 'full'

export interface VerifyOptions {
  variant?: PipelineVariant
  language_hint?: string | null
  enable_reasoning?: boolean
  store_result?: boolean
  store_image?: boolean
}

export type FeedbackResponse =
  paths['/v1/feedback']['post']['responses'][200]['content']['application/json']
export type DeleteReportResponse =
  paths['/v1/reports/{report_id}']['delete']['responses'][200]['content']['application/json']

// ============================================================================
// TYPED ERROR CLASSES
// ============================================================================

export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public code: string,
    public details?: unknown,
    public requestId?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export class UnauthorizedError extends ApiError {
  constructor(message = 'Unauthorized: Missing or invalid API key.', details?: unknown, requestId?: string) {
    super(message, 401, 'UNAUTHORIZED', details, requestId)
    this.name = 'UnauthorizedError'
  }
}

export class PayloadTooLargeError extends ApiError {
  constructor(
    message = 'Payload Too Large: Text or image payload exceeds configured server limits.',
    details?: unknown,
    requestId?: string,
  ) {
    super(message, 413, 'PAYLOAD_TOO_LARGE', details, requestId)
    this.name = 'PayloadTooLargeError'
  }
}

export class ValidationError extends ApiError {
  constructor(
    message = 'Validation Error: Request payload failed schema constraints.',
    details?: unknown,
    requestId?: string,
  ) {
    super(message, 422, 'VALIDATION_ERROR', details, requestId)
    this.name = 'ValidationError'
  }
}

export class RateLimitError extends ApiError {
  constructor(
    message = 'Rate Limit Exceeded: Please pause momentarily before sending more verification requests.',
    details?: unknown,
    requestId?: string,
  ) {
    super(message, 429, 'RATE_LIMIT_EXCEEDED', details, requestId)
    this.name = 'RateLimitError'
  }
}

export class ServerError extends ApiError {
  constructor(
    message = 'Server Error: The verification engine encountered an unexpected internal condition.',
    statusCode = 500,
    details?: unknown,
    requestId?: string,
  ) {
    super(message, statusCode, 'SERVER_ERROR', details, requestId)
    this.name = 'ServerError'
  }
}

export class NetworkError extends ApiError {
  constructor(
    message = 'Network Error: Unable to establish connection with TrustNet API server.',
    details?: unknown,
  ) {
    super(message, 0, 'NETWORK_ERROR', details)
    this.name = 'NetworkError'
  }
}

// ============================================================================
// RESPONSE ERROR PARSING HELPER
// ============================================================================

async function parseResponseError(resp: Response): Promise<ApiError> {
  const status = resp.status
  let code = 'UNKNOWN_ERROR'
  let message = resp.statusText || `Request failed with HTTP status ${status}`
  let details: unknown = undefined
  const requestId = resp.headers.get('x-request-id') || undefined

  try {
    const json = await resp.json()
    if (json && typeof json === 'object') {
      details = json
      if ('code' in json && typeof json.code === 'string') {
        code = json.code
      }
      if ('message' in json && typeof json.message === 'string') {
        message = json.message
      } else if ('detail' in json) {
        if (typeof json.detail === 'string') {
          message = json.detail
        } else {
          message = JSON.stringify(json.detail)
        }
      }
    }
  } catch {
    // If response body is non-JSON (e.g. gateway HTML error), fallback to statusText
  }

  if (status === 401) {
    return new UnauthorizedError(message, details, requestId)
  }
  if (status === 413 || code === 'TEXT_TOO_LONG' || code === 'IMAGE_TOO_LARGE') {
    return new PayloadTooLargeError(message, details, requestId)
  }
  if (status === 422) {
    return new ValidationError(message, details, requestId)
  }
  if (status === 429) {
    return new RateLimitError(message, details, requestId)
  }
  if (status >= 500) {
    return new ServerError(message, status, details, requestId)
  }

  return new ApiError(message, status, code, details, requestId)
}

// ============================================================================
// API CLIENT IMPLEMENTATION
// ============================================================================

export class TrustNetClient {
  private baseUrl: string

  constructor(baseUrl: string = '') {
    this.baseUrl = baseUrl
  }

  /**
   * Submits a factual verification job with message text and an optional image file.
   * Dispatched as multipart/form-data.
   */
  async verify(
    text: string,
    imageFile?: File | Blob | null,
    options?: VerifyOptions,
  ): Promise<JobResponse> {
    const formData = new FormData()
    if (text) {
      formData.append('text', text)
    }
    if (imageFile) {
      const filename = imageFile instanceof File ? imageFile.name : 'upload.png'
      formData.append('image', imageFile, filename)
    }
    if (options) {
      formData.append('options', JSON.stringify(options))
    }

    try {
      const resp = await fetch(`${this.baseUrl}/v1/verify`, {
        method: 'POST',
        body: formData,
      })

      if (!resp.ok) {
        throw await parseResponseError(resp)
      }

      return (await resp.json()) as JobResponse
    } catch (err) {
      if (err instanceof ApiError) {
        throw err
      }
      throw new NetworkError((err as Error).message, err)
    }
  }

  /**
   * Inquires the execution lifecycle status of an enqueued or running verification job.
   */
  async getJob(jobId: string): Promise<JobResponse> {
    try {
      const resp = await fetch(`${this.baseUrl}/v1/jobs/${encodeURIComponent(jobId)}`)
      if (!resp.ok) {
        throw await parseResponseError(resp)
      }
      return (await resp.json()) as JobResponse
    } catch (err) {
      if (err instanceof ApiError) {
        throw err
      }
      throw new NetworkError((err as Error).message, err)
    }
  }

  /**
   * Retrieves a completed verification report envelope including claim verdicts and evidence items.
   */
  async getReport(reportId: string): Promise<ReportEnvelope> {
    try {
      const resp = await fetch(`${this.baseUrl}/v1/reports/${encodeURIComponent(reportId)}`)
      if (!resp.ok) {
        throw await parseResponseError(resp)
      }
      return (await resp.json()) as ReportEnvelope
    } catch (err) {
      if (err instanceof ApiError) {
        throw err
      }
      throw new NetworkError((err as Error).message, err)
    }
  }

  /**
   * Permanently purges a stored verification report envelope and linked feedback.
   */
  async deleteReport(reportId: string): Promise<DeleteReportResponse> {
    try {
      const resp = await fetch(`${this.baseUrl}/v1/reports/${encodeURIComponent(reportId)}`, {
        method: 'DELETE',
      })
      if (!resp.ok) {
        throw await parseResponseError(resp)
      }
      return (await resp.json()) as DeleteReportResponse
    } catch (err) {
      if (err instanceof ApiError) {
        throw err
      }
      throw new NetworkError((err as Error).message, err)
    }
  }

  /**
   * Ingests human audit feedback on a generated claim report with mandatory consent check.
   */
  async submitFeedback(feedback: FeedbackIn): Promise<FeedbackResponse> {
    try {
      const resp = await fetch(`${this.baseUrl}/v1/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedback),
      })
      if (!resp.ok) {
        throw await parseResponseError(resp)
      }
      return (await resp.json()) as FeedbackResponse
    } catch (err) {
      if (err instanceof ApiError) {
        throw err
      }
      throw new NetworkError((err as Error).message, err)
    }
  }

  /**
   * Fetches service version and operational telemetry.
   */
  async getVersion(): Promise<{ version: string; service?: string }> {
    try {
      const resp = await fetch(`${this.baseUrl}/v1/version`)
      if (!resp.ok) {
        // Fallback to /v1/health if standalone /v1/version is not implemented by host proxy
        const healthResp = await fetch(`${this.baseUrl}/v1/health`)
        if (healthResp.ok) {
          const healthData = (await healthResp.json()) as { version?: string; service?: string }
          return {
            version: healthData.version || '0.1.0',
            service: healthData.service || 'trustnet-api',
          }
        }
        throw await parseResponseError(resp)
      }
      return (await resp.json()) as { version: string; service?: string }
    } catch (err) {
      if (err instanceof ApiError) {
        throw err
      }
      throw new NetworkError((err as Error).message, err)
    }
  }

  /**
   * Fetches system readiness and liveness health status.
   */
  async getHealth(): Promise<{ status: string; service: string; version: string; readiness: Record<string, unknown> }> {
    try {
      const resp = await fetch(`${this.baseUrl}/v1/health`)
      if (!resp.ok) {
        throw await parseResponseError(resp)
      }
      return (await resp.json()) as { status: string; service: string; version: string; readiness: Record<string, unknown> }
    } catch (err) {
      if (err instanceof ApiError) {
        throw err
      }
      throw new NetworkError((err as Error).message, err)
    }
  }
}

// Export singleton instance configured for relative path requests through Vite proxy or BFF
export const apiClient = new TrustNetClient()

export const verifyClaim = (args: {
  text: string
  image?: File | Blob | null
  options?: VerifyOptions
}): Promise<JobResponse> => apiClient.verify(args.text, args.image, args.options)

export const getJob = (jobId: string): Promise<JobResponse> => apiClient.getJob(jobId)

export const getReport = (reportId: string): Promise<ReportEnvelope> =>
  apiClient.getReport(reportId)

export const deleteReport = (reportId: string): Promise<DeleteReportResponse> =>
  apiClient.deleteReport(reportId)

export const submitFeedback = (
  reportId: string,
  feedback: {
    consent_to_store: boolean
    claim_id?: string | null
    suggested_verdict?: string | null
    comment?: string | null
  },
): Promise<FeedbackResponse> =>
  apiClient.submitFeedback({
    report_id: reportId,
    ...feedback,
  })

export const getVersion = (): Promise<{ version: string; service?: string }> =>
  apiClient.getVersion()

export const getHealth = (): Promise<{
  status: string
  service: string
  version: string
  readiness: Record<string, unknown>
}> => apiClient.getHealth()
