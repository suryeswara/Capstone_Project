import { X, LogOut, ShieldCheck, CheckCircle2, XCircle, Database, Calendar, Award } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { Button } from '@/components/ui/Primitives'

export function UserProfileModal() {
  const { isProfileModalOpen, setProfileModalOpen, user, userProfile, logoutUser, history } = useAppStore()

  if (!isProfileModalOpen || !user) return null

  const initials = user.fullName
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
      <div className="relative w-full max-w-lg overflow-hidden rounded-2xl border border-mist-200 bg-white shadow-2xl transition-all">
        {/* Header Background Banner */}
        <div className="h-24 bg-gradient-to-r from-ink-950 via-clinical-900 to-ink-900 p-4">
          <button
            onClick={() => setProfileModalOpen(false)}
            className="absolute right-4 top-4 rounded-lg bg-black/20 p-1 text-white/80 hover:bg-black/40 hover:text-white"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Profile Info Overlay */}
        <div className="px-6 pb-6 pt-0">
          <div className="relative -top-10 mb-[-24px] flex items-end justify-between">
            <div className="flex h-20 w-20 items-center justify-center rounded-2xl border-4 border-white bg-trust-500 text-2xl font-bold text-white shadow-lg">
              {initials}
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1 rounded-full bg-clinical-50 px-3 py-1 text-xs font-semibold text-clinical-700 border border-clinical-200">
                <Award className="h-3.5 w-3.5" />
                {user.role.toUpperCase()}
              </span>
            </div>
          </div>

          <div className="mt-4">
            <h2 className="text-xl font-bold text-ink-950">{user.fullName}</h2>
            <p className="text-xs text-ink-500">{user.email}</p>
            <div className="mt-2 flex items-center gap-4 text-xs text-ink-600">
              <span className="flex items-center gap-1">
                <Calendar className="h-3.5 w-3.5 text-ink-400" />
                Member since {userProfile?.memberSince || 'August 2026'}
              </span>
              <span className="flex items-center gap-1">
                <Database className="h-3.5 w-3.5 text-trust-500" />
                PostgreSQL Synced
              </span>
            </div>
          </div>

          {/* Account Statistics Grid */}
          <div className="mt-6 grid grid-cols-3 gap-3">
            <div className="rounded-xl border border-mist-200 bg-mist-50/50 p-3 text-center">
              <p className="text-xs font-medium text-ink-500">Submitted</p>
              <p className="text-lg font-bold text-ink-950">
                {userProfile?.totalVerificationsSubmitted ?? history.length}
              </p>
            </div>
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-3 text-center">
              <p className="text-xs font-medium text-emerald-700">Supported</p>
              <p className="text-lg font-bold text-emerald-700">
                {userProfile?.totalClaimsSupported ?? history.filter((h) => h.verdict === 'Supported').length}
              </p>
            </div>
            <div className="rounded-xl border border-red-200 bg-red-50/40 p-3 text-center">
              <p className="text-xs font-medium text-red-700">Contradicted</p>
              <p className="text-lg font-bold text-red-700">
                {userProfile?.totalClaimsContradicted ?? history.filter((h) => h.verdict === 'Contradicted').length}
              </p>
            </div>
          </div>

          {/* Recent DB Saved Claims */}
          <div className="mt-6">
            <h3 className="mb-2.5 text-xs font-bold uppercase tracking-wider text-ink-400">
              Your Verification History (DB Synced)
            </h3>
            <div className="max-h-48 space-y-2 overflow-y-auto pr-1">
              {history.length === 0 ? (
                <p className="py-4 text-center text-xs text-ink-400">No verifications submitted yet.</p>
              ) : (
                history.slice(0, 5).map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between rounded-xl border border-mist-200 bg-white p-3 shadow-xs hover:border-ink-300"
                  >
                    <div className="min-w-0 pr-3">
                      <p className="truncate text-xs font-medium text-ink-900">{item.claim}</p>
                      <p className="text-[10px] text-ink-400">{item.disease}</p>
                    </div>
                    <span
                      className={`inline-flex flex-none items-center gap-1 rounded-md px-2 py-0.5 text-[10px] font-semibold ${
                        item.verdict === 'Supported'
                          ? 'bg-emerald-50 text-emerald-700'
                          : item.verdict === 'Contradicted'
                          ? 'bg-red-50 text-red-700'
                          : 'bg-amber-50 text-amber-700'
                      }`}
                    >
                      {item.verdict === 'Supported' ? (
                        <CheckCircle2 className="h-3 w-3" />
                      ) : (
                        <XCircle className="h-3 w-3" />
                      )}
                      {item.verdict}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Sign Out Button */}
          <div className="mt-6 border-t border-mist-100 pt-4 flex justify-between items-center">
            <span className="text-xs text-ink-400">MedVerify AI Database Linked</span>
            <Button
              variant="secondary"
              onClick={() => logoutUser()}
              className="border-red-200 text-red-600 hover:bg-red-50 hover:text-red-700"
            >
              <LogOut className="mr-2 h-4 w-4" />
              Sign Out
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
