import { http, HttpResponse } from 'msw'
import type { components } from '../api/types'

type JobResponse = components['schemas']['JobResponse']
type ReportEnvelope = components['schemas']['ReportEnvelope']

export const mockReportVerified: ReportEnvelope = {
  report_id: 'rep-verified-123',
  pipeline_version: 'git-commit-abc1234',
  model_versions: {
    text: 'mdeberta-v3-base',
    retriever: 'bge-large-en-v1.5',
    fusion: 'cross-attention-v1',
  },
  config_hash: 'a1b2c3d4e5f6',
  variant: 'full',
  timings: {
    extraction: 0.12,
    retrieval: 0.35,
    multimodal_fusion: 0.22,
    reasoning: 0.15,
  },
  degraded: [],
  original_text: 'WHO confirms booster shot reduces hospitalization',
  claims: [
    {
      claim: {
        id: 'c-1',
        text: 'WHO confirms booster shot reduces hospitalization',
      },
      verdict: 'verified',
      confidence: 0.94,
      abstained: false,
      share_advice: 'safe_to_share',
      evidence: [
        {
          doc_id: 'doc-1',
          text: 'World Health Organization clinical trials conclude booster vaccination significantly cuts severe hospitalizations.',
          source: 'WHO Official Guidelines',
          source_type: 'factcheck',
          date: '2026-02-15',
          url: 'https://who.int/news/boosters',
          score: 0.92,
        },
      ],
      conflict: {
        has_conflict: false,
        explanation: '',
        supporting: [
          {
            doc_id: 'doc-1',
            text: 'World Health Organization clinical trials conclude booster vaccination significantly cuts severe hospitalizations.',
            source: 'WHO Official Guidelines',
            source_type: 'factcheck',
            date: '2026-02-15',
            url: 'https://who.int/news/boosters',
            score: 0.92,
          },
        ],
        contradicting: [],
      },
      source_weights: [
        {
          source_type: 'factcheck',
          weight: 0.95,
          rationale: 'Primary accredited international health organization document.',
        },
      ],
      temporal: {
        is_stale: false,
        claim_time_expressions: ['2026'],
        newest_evidence_date: '2026-02-15',
        oldest_evidence_date: '2026-01-10',
        explanation: 'Evidence is current and directly addresses the 2026 timeframe.',
      },
      gaps: [],
      manipulation: [],
      counterfactuals: [
        {
          change_description: 'If evidence indicated no reduction in hospitalizations',
          original_verdict: 'verified',
          new_verdict: 'contradicted',
          flipped: true,
        },
      ],
    },
  ],
}

export const mockReportAbstained: ReportEnvelope = {
  report_id: 'rep-abstained-456',
  pipeline_version: 'git-commit-abc1234',
  model_versions: {
    text: 'mdeberta-v3-base',
  },
  config_hash: 'a1b2c3d4e5f6',
  variant: 'full',
  timings: {
    extraction: 0.05,
    retrieval: 0.45,
    reasoning: 0.10,
  },
  degraded: ['image_processing'],
  original_text: 'Secret ingredient in local water causes spontaneous levitation',
  claims: [
    {
      claim: {
        id: 'c-2',
        text: 'Secret ingredient in local water causes spontaneous levitation',
      },
      verdict: 'insufficient_evidence',
      confidence: 0.28,
      abstained: true,
      abstention_reason: 'low_confidence_and_out_of_domain',
      share_advice: 'do_not_share_as_confirmed',
      evidence: [],
      conflict: {
        has_conflict: true,
        explanation: 'Sensational web blogs dispute mainstream physical physics publications.',
        supporting: [],
        contradicting: [
          {
            doc_id: 'doc-phys-1',
            text: 'Physical dynamics prohibit spontaneous human levitation through municipal tap water.',
            source: 'Journal of Physics',
            source_type: 'academic',
            date: '2025-11-01',
            url: 'https://academic.example/phys',
            score: 0.88,
          },
        ],
      },
      source_weights: [
        {
          source_type: 'academic',
          weight: 0.9,
          rationale: 'Peer-reviewed scholarly literature.',
        },
      ],
      temporal: {
        is_stale: true,
        claim_time_expressions: ['secret'],
        newest_evidence_date: '2025-11-01',
        oldest_evidence_date: '2024-05-10',
        explanation: 'No recent municipal records support this.',
      },
      gaps: [
        {
          code: 'GAP_NO_CHEMICAL_TEST',
          description: 'No certified lab chemical analysis was provided for municipal water sample.',
          how_to_check: 'Request certified chemical assay from regional water board.',
        },
      ],
      manipulation: [
        {
          name: 'conspiratorial_tone',
          confidence: 0.89,
          span: 'Secret ingredient in local water',
        },
      ],
      counterfactuals: [
        {
          change_description: 'If verified peer-reviewed lab test confirms water composition anomaly',
          original_verdict: 'abstained',
          new_verdict: 'insufficient',
          flipped: false,
        },
      ],
    },
  ],
}

export const handlers = [
  // POST /v1/verify
  http.post('*/v1/verify', async () => {
    const job: JobResponse = {
      job_id: 'job-mock-test-123',
      status: 'queued',
      stage: 'queued',
      progress: 0,
      created_at: new Date().toISOString(),
    }
    return HttpResponse.json(job, { status: 202 })
  }),

  // GET /v1/jobs/:id
  http.get('*/v1/jobs/:id', ({ params }) => {
    const id = params.id
    if (id === 'job-done') {
      const job: JobResponse = {
        job_id: 'job-done',
        status: 'done',
        stage: 'completed',
        progress: 1.0,
        report: mockReportVerified,
        created_at: new Date().toISOString(),
      }
      return HttpResponse.json(job)
    }

    if (id === 'job-failed') {
      const job: JobResponse = {
        job_id: 'job-failed',
        status: 'failed',
        error: 'Failed to complete multimodal grounding.',
        progress: 0.5,
        created_at: new Date().toISOString(),
      }
      return HttpResponse.json(job)
    }

    // Default running
    const job: JobResponse = {
      job_id: String(id),
      status: 'running',
      stage: 'evidence_retrieval',
      progress: 0.65,
      created_at: new Date().toISOString(),
    }
    return HttpResponse.json(job)
  }),

  // GET /v1/reports/:id
  http.get('*/v1/reports/:id', ({ params }) => {
    const id = params.id
    if (id === 'rep-abstained-456') {
      return HttpResponse.json(mockReportAbstained)
    }
    return HttpResponse.json(mockReportVerified)
  }),

  // DELETE /v1/reports/:id
  http.delete('*/v1/reports/:id', ({ params }) => {
    return HttpResponse.json({
      status: 'deleted',
      report_id: params.id,
      purged_at: new Date().toISOString(),
    })
  }),

  // POST /v1/feedback
  http.post('*/v1/feedback', async () => {
    return HttpResponse.json({
      status: 'received',
      feedback_id: 'fb-test-999',
    })
  }),

  // GET /v1/version
  http.get('*/v1/version', () => {
    return HttpResponse.json({
      version: '0.1.0-test',
      service: 'trustnet-api',
      status: 'operational',
    })
  }),

  // GET /v1/health
  http.get('*/v1/health', () => {
    return HttpResponse.json({
      status: 'healthy',
      service: 'trustnet-api',
      version: '0.1.0-test',
      readiness: { models: true, database: true },
    })
  }),
]
