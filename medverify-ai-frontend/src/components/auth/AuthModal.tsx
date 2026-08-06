import { useState } from 'react'
import { X, Lock, Mail, User, ShieldCheck, AlertCircle, Loader2 } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { Button } from '@/components/ui/Primitives'

export function AuthModal() {
  const { isAuthModalOpen, setAuthModalOpen, loginUser, registerUser, authError } = useAppStore()
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [loading, setLoading] = useState(false)

  // Form State
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [role, setRole] = useState('researcher')

  if (!isAuthModalOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      if (mode === 'login') {
        await loginUser(email, password)
      } else {
        await registerUser(email, password, fullName, role)
      }
    } catch {
      // Error handled by store
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
      <div className="relative w-full max-w-md overflow-hidden rounded-2xl border border-mist-200 bg-white p-6 shadow-2xl transition-all">
        {/* Close Button */}
        <button
          onClick={() => setAuthModalOpen(false)}
          className="absolute right-4 top-4 rounded-lg p-1 text-ink-400 hover:bg-mist-100 hover:text-ink-700"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Modal Header */}
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-ink-950 text-clinical-400 shadow-md">
            <ShieldCheck className="h-5 w-5" strokeWidth={2.25} />
          </div>
          <div>
            <h2 className="text-lg font-bold text-ink-950">
              {mode === 'login' ? 'Sign in to MedVerify AI' : 'Create Researcher Account'}
            </h2>
            <p className="text-xs text-ink-500">
              {mode === 'login'
                ? 'Access your saved claim verifications and research data'
                : 'Join the clinical evidence verification platform'}
            </p>
          </div>
        </div>

        {/* Error Alert */}
        {authError && (
          <div className="mb-4 flex items-center gap-2.5 rounded-xl border border-red-200 bg-red-50 p-3 text-xs font-medium text-red-700">
            <AlertCircle className="h-4 w-4 flex-none" />
            <span>{authError}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === 'register' && (
            <div>
              <label className="mb-1 block text-xs font-semibold text-ink-700">Full Name</label>
              <div className="relative">
                <User className="absolute left-3 top-2.5 h-4 w-4 text-ink-400" />
                <input
                  type="text"
                  required
                  placeholder="Dr. Sarah Lin"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="w-full rounded-xl border border-mist-200 bg-mist-50/50 py-2.5 pl-9 pr-3 text-sm text-ink-900 placeholder:text-ink-400 focus:border-ink-950 focus:bg-white focus:outline-none"
                />
              </div>
            </div>
          )}

          <div>
            <label className="mb-1 block text-xs font-semibold text-ink-700">Email Address</label>
            <div className="relative">
              <Mail className="absolute left-3 top-2.5 h-4 w-4 text-ink-400" />
              <input
                type="email"
                required
                placeholder="researcher@medverify.ai"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-xl border border-mist-200 bg-mist-50/50 py-2.5 pl-9 pr-3 text-sm text-ink-900 placeholder:text-ink-400 focus:border-ink-950 focus:bg-white focus:outline-none"
              />
            </div>
          </div>

          <div>
            <label className="mb-1 block text-xs font-semibold text-ink-700">Password</label>
            <div className="relative">
              <Lock className="absolute left-3 top-2.5 h-4 w-4 text-ink-400" />
              <input
                type="password"
                required
                minLength={6}
                placeholder="••••••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-xl border border-mist-200 bg-mist-50/50 py-2.5 pl-9 pr-3 text-sm text-ink-900 placeholder:text-ink-400 focus:border-ink-950 focus:bg-white focus:outline-none"
              />
            </div>
          </div>

          {mode === 'register' && (
            <div>
              <label className="mb-1 block text-xs font-semibold text-ink-700">Account Role</label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value)}
                className="w-full rounded-xl border border-mist-200 bg-mist-50/50 py-2.5 px-3 text-sm text-ink-900 focus:border-ink-950 focus:bg-white focus:outline-none"
              >
                <option value="researcher">Clinical Researcher</option>
                <option value="reviewer">Medical Reviewer</option>
                <option value="user">General User</option>
              </select>
            </div>
          )}

          <Button type="submit" disabled={loading} className="w-full py-2.5 font-semibold">
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : mode === 'login' ? (
              'Sign In'
            ) : (
              'Create Account'
            )}
          </Button>
        </form>

        {/* Toggle Mode Footer */}
        <div className="mt-5 border-t border-mist-100 pt-4 text-center">
          <p className="text-xs text-ink-500">
            {mode === 'login' ? "Don't have an account?" : 'Already have an account?'}
            <button
              onClick={() => setMode(mode === 'login' ? 'register' : 'login')}
              className="ml-1 font-semibold text-ink-950 underline hover:text-clinical-600"
            >
              {mode === 'login' ? 'Create Account' : 'Sign In'}
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}
