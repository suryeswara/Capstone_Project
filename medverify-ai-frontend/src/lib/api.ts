import type { VerificationRecord, VerificationMode, PipelineStage } from '@/types'

const API_BASE = 'http://localhost:8000/api'

// ---------------------------------------------------------------------------
// AUTH INTERFACES & FUNCTIONS
// ---------------------------------------------------------------------------

export interface UserDTO {
  id: string
  email: string
  fullName: string
  role: string
  createdAt: string
}

export interface AuthResponse {
  accessToken: string
  tokenType: string
  user: UserDTO
}

export interface UserProfileResponse {
  user: UserDTO
  totalVerificationsSubmitted: number
  totalClaimsSupported: number
  totalClaimsContradicted: number
  memberSince: string
}

export async function loginApi(email: string, password: string): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Login failed')
  }
  return res.json()
}

export async function registerApi(email: string, password: string, fullName: string, role: string = 'user'): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, fullName, role })
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Registration failed')
  }
  return res.json()
}

export async function fetchProfileApi(token: string): Promise<UserProfileResponse> {
  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` }
  })
  if (!res.ok) {
    throw new Error('Failed to fetch profile')
  }
  return res.json()
}

export async function fetchUserHistoryApi(token: string): Promise<VerificationRecord[]> {
  const res = await fetch(`${API_BASE}/auth/history`, {
    headers: { Authorization: `Bearer ${token}` }
  })
  if (!res.ok) {
    throw new Error('Failed to fetch user history')
  }
  const statusList = await res.json()
  
  // Fetch detailed reports for completed verifications
  const reports = await Promise.all(
    statusList.map(async (item: any) => {
      try {
        return await fetchReportApi(item.verificationId)
      } catch {
        return null
      }
    })
  )
  
  return (reports.filter(Boolean) as VerificationRecord[]).filter((r) => r.status === 'COMPLETED')
}

// ---------------------------------------------------------------------------
// CLAIM & VERIFICATION INTERFACES
// ---------------------------------------------------------------------------

export interface SubmitClaimResponse {
  claimId: string
  verificationId: string
  status: string
  submittedAt: string
  pollUrl: string
}

export interface VerificationStatusResponse {
  verificationId: string
  claimId: string
  rawText: string
  status: string
  progressPercentage: number
  currentStepLabel?: string
  updatedAt: string
  isTerminal: boolean
}

export async function submitClaimApi(rawText: string, mode: VerificationMode = 'static', token?: string): Promise<SubmitClaimResponse> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const res = await fetch(`${API_BASE}/claims`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ rawText, mode })
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(typeof err.detail === 'string' ? err.detail : `Failed to submit claim: ${res.statusText}`)
  }
  return res.json()
}

export async function submitClaimImageApi(file: File, mode: VerificationMode = 'static', token?: string): Promise<SubmitClaimResponse> {
  const headers: Record<string, string> = {}
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const formData = new FormData()
  formData.append('file', file)
  formData.append('mode', mode)

  const res = await fetch(`${API_BASE}/claims/image`, {
    method: 'POST',
    headers,
    body: formData
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(typeof err.detail === 'string' ? err.detail : `Failed to upload claim image: ${res.statusText}`)
  }
  return res.json()
}

export async function pollStatusApi(verificationId: string): Promise<VerificationStatusResponse> {

  const res = await fetch(`${API_BASE}/verifications/${verificationId}`)
  if (!res.ok) {
    throw new Error(`Failed to poll status: ${res.statusText}`)
  }
  return res.json()
}

export async function fetchReportApi(verificationId: string): Promise<VerificationRecord> {
  const res = await fetch(`${API_BASE}/verifications/${verificationId}/report`)
  if (!res.ok) {
    throw new Error(`Failed to fetch report: ${res.statusText}`)
  }
  const data = await res.json()

  const evidence = (data.evidence || []).map((e: any) => ({
    id: e.id,
    title: e.title,
    source: e.sourceType,
    authors: e.authors || undefined,
    year: e.pubYear ?? undefined,
    pubYear: e.pubYear ?? undefined,
    doi: e.doi || undefined,
    pmid: e.pmid || undefined,
    pool: e.pool || undefined,
    similarity: e.similarity ?? undefined,
    relevanceScore: e.relevanceScore ?? 0,
    reliabilityScore: Math.round((e.relevanceScore ?? 0) * 100),
    stance: e.stance || 'neutral',
    stanceProbs: e.stanceProbs || undefined,
    abstract: e.abstractChunk || undefined,
    url: e.url || undefined,
  }))

  // Real per-year stance counts from the retrieved evidence (no placeholder data).
  const byYear = new Map<number, { year: number; supporting: number; contradicting: number; neutral: number }>()
  for (const e of evidence) {
    if (!e.year) continue
    const row = byYear.get(e.year) ?? { year: e.year, supporting: 0, contradicting: 0, neutral: 0 }
    row[e.stance as 'supporting' | 'contradicting' | 'neutral'] += 1
    byYear.set(e.year, row)
  }

  const summary = data.explanationSummary
  return {
    id: data.verificationId,
    claim: data.rawText,
    disease: 'General Medicine',
    submittedAt: data.completedAt || new Date().toISOString(),
    verdict: data.verdict || 'Insufficient Evidence',
    systemConfidence: data.systemConfidence ?? undefined,
    credibility: data.credibility || { overall: 0, evidenceConfidence: 0, consensusConfidence: 0, sourceQuality: 0, faithfulnessConfidence: null },
    evidence,
    consensusTimeline: [...byYear.values()].sort((a, b) => a.year - b.year),
    explanation: (data.explanation || []).map((ex: any) => ({
      id: ex.sentenceId,
      text: ex.text,
      status: ex.status,
      confidence: Math.round((ex.nliConfidence ?? 0) * 100),
      evidenceIds: ex.citedEvidenceIds || [],
    })),
    modelVersion: data.versionMetadata?.modelVersion || '',
    status: data.status,
    mode: data.mode || undefined,
    modeEffective: data.modeEffective || undefined,
    probabilities: data.probabilities || undefined,
    needsReview: data.needsReview ?? undefined,
    reviewThreshold: data.reviewThreshold ?? undefined,
    gate: data.gate || null,
    notes: data.notes || [],
    summary: summary
      ? { writer: summary.writer, why: summary.why, caveats: summary.caveats || [], nextSteps: summary.nextSteps || [], strength: summary.strength }
      : null,
    suggestions: data.suggestions || [],
    disclaimer: data.disclaimer || null,
  }
}

export function subscribeToVerificationStream(
  verificationId: string,
  onUpdate: (data: VerificationStatusResponse) => void,
  onError?: (err: any) => void
): () => void {
  const eventSource = new EventSource(`${API_BASE}/verifications/${verificationId}/stream`)

  eventSource.addEventListener('status_update', (event) => {
    try {
      const data: VerificationStatusResponse = JSON.parse(event.data)
      onUpdate(data)
      if (data.isTerminal) {
        eventSource.close()
      }
    } catch (e) {
      if (onError) onError(e)
    }
  })

  eventSource.onerror = (err) => {
    if (onError) onError(err)
    eventSource.close()
  }

  return () => {
    eventSource.close()
  }
}
