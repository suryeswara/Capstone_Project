import { create } from 'zustand'
import type { VerificationRecord, PipelineStage, DiseaseCategory, VerificationMode } from '@/types'
import { freshStages } from '@/lib/pipeline'
import {
  submitClaimApi,
  submitClaimImageApi,
  pollStatusApi,
  fetchReportApi,
  loginApi,
  registerApi,
  fetchProfileApi,
  fetchUserHistoryApi,
  type UserDTO,
  type UserProfileResponse
} from '@/lib/api'


interface AppState {
  // Claim & Verification State
  history: VerificationRecord[]
  activeClaim: string
  activeDisease: DiseaseCategory | null
  stages: PipelineStage[]
  isRunning: boolean
  result: VerificationRecord | null
  error: string | null
  verificationMode: VerificationMode

  // Authentication State
  user: UserDTO | null
  token: string | null
  isAuthenticated: boolean
  userProfile: UserProfileResponse | null
  isAuthModalOpen: boolean
  isProfileModalOpen: boolean
  authError: string | null

  // Actions
  setActiveClaim: (c: string) => void
  toggleBookmark: (id: string) => void
  setVerificationMode: (m: VerificationMode) => void
  runVerification: (claim: string) => Promise<void>
  runImageVerification: (file: File) => Promise<void>
  reset: () => void


  // Auth Actions
  setAuthModalOpen: (open: boolean) => void
  setProfileModalOpen: (open: boolean) => void
  loginUser: (email: string, password: string) => Promise<void>
  registerUser: (email: string, password: string, fullName: string, role?: string) => Promise<void>
  logoutUser: () => void
  loadStoredAuth: () => Promise<void>
}

const POLL_INTERVAL_MS = 500
const POLL_TIMEOUT_MS = 180_000 // live/hybrid runs include PubMed + explanation; allow up to 3 minutes

type SetFn = (fn: Partial<AppState> | ((s: AppState) => Partial<AppState>)) => void

/** Poll until the backend reports a terminal state, then load the real report. Never fabricates a result. */
async function trackVerification(verificationId: string, set: SetFn): Promise<void> {
  const started = Date.now()
  let label = ''
  const nStages = freshStages().length
  for (;;) {
    const st = await pollStatusApi(verificationId)
    label = st.currentStepLabel || label
    const activeIdx = Math.min(nStages - 1, Math.floor((st.progressPercentage / 100) * nStages))
    set((s) => ({
      activeClaim: st.rawText || s.activeClaim,
      stages: s.stages.map((x, i) =>
        i === activeIdx ? { ...x, status: 'active', detail: st.currentStepLabel } : i < activeIdx ? { ...x, status: 'complete' } : x
      ),
    }))
    if (st.isTerminal) break
    if (Date.now() - started > POLL_TIMEOUT_MS) throw new Error('The verification is taking too long. Please try again.')
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS))
  }
  const report = await fetchReportApi(verificationId)
  if (report.status === 'FAILED') throw new Error(label || 'The verification failed.')
  // Rejected (not medical) and refused (safety) claims carry no verdict: show the notice, keep them out of history.
  const result = report.status === 'COMPLETED' ? report : { ...report, statusLabel: label }
  set((s) => ({
    isRunning: false,
    activeClaim: result.claim || s.activeClaim,
    stages: s.stages.map((x) => ({ ...x, status: 'complete' })),
    result,
    history: report.status === 'COMPLETED' ? [result, ...s.history] : s.history,
  }))
}

function failRun(err: unknown, set: SetFn): void {
  const message = err instanceof Error ? err.message : 'The verification service is unavailable.'
  console.error('[verify] failed:', err)
  set({ isRunning: false, error: message })
}

export const useAppStore = create<AppState>((set, get) => ({
  history: [],
  error: null,
  verificationMode: 'static',
  activeClaim: '',
  activeDisease: null,
  stages: freshStages(),
  isRunning: false,
  result: null,

  // Auth Initial State
  user: null,
  token: null,
  isAuthenticated: false,
  userProfile: null,
  isAuthModalOpen: false,
  isProfileModalOpen: false,
  authError: null,

  setActiveClaim: (c) => set({ activeClaim: c }),

  toggleBookmark: (id) =>
    set((s) => ({
      history: s.history.map((h) => (h.id === id ? { ...h, bookmarked: !h.bookmarked } : h)),
    })),

  // Auth Modal Toggles
  setAuthModalOpen: (open) => set({ isAuthModalOpen: open, authError: null }),
  setProfileModalOpen: (open) => {
    set({ isProfileModalOpen: open })
    if (open && get().token) {
      // Refresh profile data from DB when modal opens
      get().loadStoredAuth()
    }
  },

  loginUser: async (email, password) => {
    set({ authError: null })
    try {
      const auth = await loginApi(email, password)
      localStorage.setItem('medverify_jwt_token', auth.accessToken)
      localStorage.setItem('medverify_user', JSON.stringify(auth.user))

      set({
        user: auth.user,
        token: auth.accessToken,
        isAuthenticated: true,
        isAuthModalOpen: false,
        authError: null,
      })

      // Fetch full profile stats and history from DB
      const profile = await fetchProfileApi(auth.accessToken)
      const userHistory = await fetchUserHistoryApi(auth.accessToken)
      set((s) => ({
        userProfile: profile,
        history: userHistory.length > 0 ? userHistory : s.history
      }))
    } catch (err: any) {
      set({ authError: err.message || 'Login failed' })
      throw err
    }
  },

  registerUser: async (email, password, fullName, role = 'user') => {
    set({ authError: null })
    try {
      const auth = await registerApi(email, password, fullName, role)
      localStorage.setItem('medverify_jwt_token', auth.accessToken)
      localStorage.setItem('medverify_user', JSON.stringify(auth.user))

      set({
        user: auth.user,
        token: auth.accessToken,
        isAuthenticated: true,
        isAuthModalOpen: false,
        authError: null,
      })

      const profile = await fetchProfileApi(auth.accessToken)
      set({ userProfile: profile })
    } catch (err: any) {
      set({ authError: err.message || 'Registration failed' })
      throw err
    }
  },

  logoutUser: () => {
    localStorage.removeItem('medverify_jwt_token')
    localStorage.removeItem('medverify_user')
    set({
      user: null,
      token: null,
      isAuthenticated: false,
      userProfile: null,
      isProfileModalOpen: false,
      isAuthModalOpen: false,
      authError: null,
    })
  },

  loadStoredAuth: async () => {
    const token = localStorage.getItem('medverify_jwt_token')
    const storedUserStr = localStorage.getItem('medverify_user')
    if (!token) return

    try {
      const user: UserDTO = storedUserStr ? JSON.parse(storedUserStr) : null
      set({ token, user, isAuthenticated: !!token })

      const profile = await fetchProfileApi(token)
      const userHistory = await fetchUserHistoryApi(token)

      set((s) => ({
        userProfile: profile,
        user: profile.user,
        history: userHistory.length > 0 ? userHistory : s.history
      }))
    } catch (err) {
      console.warn("Stored token invalid or expired. Clearing local auth:", err)
      get().logoutUser()
    }
  },

  setVerificationMode: (m) => set({ verificationMode: m }),

  runVerification: async (claim: string) => {
    const { token, verificationMode } = get()
    set({ isRunning: true, result: null, error: null, stages: freshStages(), activeClaim: claim })
    try {
      const { verificationId } = await submitClaimApi(claim, verificationMode, token || undefined)
      await trackVerification(verificationId, set)
    } catch (err) {
      failRun(err, set)
    }
  },

  runImageVerification: async (file: File) => {
    const { token, verificationMode } = get()
    set({ isRunning: true, result: null, error: null, stages: freshStages(), activeClaim: `[Image: ${file.name}]` })
    try {
      const { verificationId } = await submitClaimImageApi(file, verificationMode, token || undefined)
      await trackVerification(verificationId, set)
    } catch (err) {
      failRun(err, set)
    }
  },

  reset: () => set({ activeClaim: '', stages: freshStages(), result: null, isRunning: false, error: null }),
}))

