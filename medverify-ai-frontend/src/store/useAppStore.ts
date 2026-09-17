import { create } from 'zustand'
import type { VerificationRecord, PipelineStage, DiseaseCategory } from '@/types'
import { VERIFICATIONS } from '@/lib/mockData'
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

function pickMockResult(claim: string): VerificationRecord {
  const lower = claim.toLowerCase()
  const byKeyword = VERIFICATIONS.find((v) =>
    lower.includes('statin') || lower.includes('heart') || lower.includes('aspirin')
      ? v.disease === 'Cardiovascular Disease'
      : lower.includes('diabetes') || lower.includes('fasting') || lower.includes('insulin')
      ? v.disease === 'Diabetes'
      : lower.includes('vaccine') || lower.includes('vaccination') || lower.includes('mmr') || lower.includes('flu')
      ? v.disease === 'Vaccination'
      : false
  )
  const base = byKeyword ?? VERIFICATIONS[Math.floor(Math.random() * VERIFICATIONS.length)]
  return {
    ...base,
    id: `ver-${Date.now()}`,
    claim,
    submittedAt: new Date().toISOString(),
  }
}

export const useAppStore = create<AppState>((set, get) => ({
  history: VERIFICATIONS,
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

  runVerification: async (claim: string) => {
    const stages = freshStages()
    set({ isRunning: true, result: null, stages, activeClaim: claim })
    const { token } = get()

    try {
      // 1. Submit claim to backend API (passes JWT token if logged in)
      const { verificationId } = await submitClaimApi(claim, token || undefined)

      // 2. Poll backend status
      let isDone = false
      let attempts = 0
      while (!isDone && attempts < 35) {
        await new Promise((r) => setTimeout(r, 400))
        attempts++

        const statusRes = await pollStatusApi(verificationId)

        // Map backend progress percentage to frontend 8 stages
        const activeIdx = Math.min(
          stages.length - 1,
          Math.floor((statusRes.progressPercentage / 100) * stages.length)
        )

        set((s) => ({
          stages: s.stages.map((st, idx) =>
            idx === activeIdx ? { ...st, status: 'active', detail: statusRes.currentStepLabel } : idx < activeIdx ? { ...st, status: 'complete' } : st
          )
        }))

        if (statusRes.isTerminal) {
          isDone = true
        }
      }

      // 3. Fetch full report payload
      const result = await fetchReportApi(verificationId)

      set((s) => ({
        isRunning: false,
        stages: s.stages.map((st) => ({ ...st, status: 'complete' })),
        result,
        history: [result, ...s.history]
      }))

    } catch (err) {
      console.warn("Backend API offline/unreachable, falling back to client simulation:", err)

      let cumulative = 0
      stages.forEach((stage, idx) => {
        cumulative += idx === 0 ? 200 : stages[idx - 1].durationMs
        setTimeout(() => {
          set((s) => ({
            stages: s.stages.map((st, i) =>
              i === idx ? { ...st, status: 'active' } : i < idx ? { ...st, status: 'complete' } : st
            ),
          }))
        }, cumulative)
      })

      const totalTime = cumulative + stages[stages.length - 1].durationMs
      setTimeout(() => {
        set((s) => ({
          stages: s.stages.map((st) => ({ ...st, status: 'complete' })),
        }))
        const result = pickMockResult(claim)
        set((s) => ({
          isRunning: false,
          result,
          history: [result, ...s.history],
        }))
      }, totalTime + 250)
    }
  },

  runImageVerification: async (file: File) => {
    const stages = freshStages()
    set({ isRunning: true, result: null, stages, activeClaim: `[Image: ${file.name}]` })
    const { token } = get()

    try {
      const { verificationId } = await submitClaimImageApi(file, token || undefined)

      let isDone = false
      let attempts = 0
      while (!isDone && attempts < 35) {
        await new Promise((r) => setTimeout(r, 400))
        attempts++

        const statusRes = await pollStatusApi(verificationId)
        const activeIdx = Math.min(
          stages.length - 1,
          Math.floor((statusRes.progressPercentage / 100) * stages.length)
        )

        set((s) => ({
          activeClaim: statusRes.rawText || s.activeClaim,
          stages: s.stages.map((st, idx) =>
            idx === activeIdx ? { ...st, status: 'active', detail: statusRes.currentStepLabel } : idx < activeIdx ? { ...st, status: 'complete' } : st
          )
        }))

        if (statusRes.isTerminal) {
          isDone = true
        }
      }

      const result = await fetchReportApi(verificationId)
      set((s) => ({
        isRunning: false,
        activeClaim: result.claim,
        stages: s.stages.map((st) => ({ ...st, status: 'complete' })),
        result,
        history: [result, ...s.history]
      }))
    } catch (err) {
      console.warn("Backend image verification failed, falling back to client verification:", err)
      get().runVerification("Drinking lemon water on an empty stomach cures type 2 diabetes.")
    }
  },

  reset: () => set({ activeClaim: '', stages: freshStages(), result: null, isRunning: false }),
}))

