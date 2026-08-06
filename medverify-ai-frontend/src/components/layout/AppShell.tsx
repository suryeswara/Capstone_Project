import { useEffect } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard,
  ShieldCheck,
  FlaskConical,
  ScaleIcon,
  Sparkles,
  Gauge,
  History,
  BarChart3,
  Settings,
  Search,
  Bell,
  LogIn,
  User,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useAppStore } from '@/store/useAppStore'
import { AuthModal } from '@/components/auth/AuthModal'
import { UserProfileModal } from '@/components/auth/UserProfileModal'

const NAV = [
  { to: '/app/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/app/verify', label: 'Verify a claim', icon: ShieldCheck },
  { to: '/app/evidence', label: 'Evidence explorer', icon: FlaskConical },
  { to: '/app/consensus', label: 'Scientific consensus', icon: ScaleIcon },
  { to: '/app/faithfulness', label: 'Faithfulness lab', icon: Sparkles },
  { to: '/app/history', label: 'History', icon: History },
  { to: '/app/analytics', label: 'Analytics', icon: BarChart3 },
  { to: '/app/admin', label: 'Admin panel', icon: Settings },
]

export function AppShell() {
  const navigate = useNavigate()
  const { isAuthenticated, user, setAuthModalOpen, setProfileModalOpen, loadStoredAuth } = useAppStore()

  useEffect(() => {
    loadStoredAuth()
  }, [])

  const initials = user?.fullName
    ? user.fullName
        .split(' ')
        .map((n) => n[0])
        .join('')
        .toUpperCase()
    : 'U'

  return (
    <div className="flex min-h-screen bg-mist-50">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 flex-col border-r border-mist-200 bg-white lg:flex">
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2 border-b border-mist-200 px-6 py-5 text-left"
        >
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink-950">
            <ShieldCheck className="h-4.5 w-4.5 text-clinical-400" strokeWidth={2.25} />
          </div>
          <span className="text-[15px] font-semibold tracking-tight text-ink-950">MedVerify AI</span>
        </button>
        <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-5">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors',
                  isActive ? 'bg-ink-950 text-white' : 'text-ink-600 hover:bg-mist-100 hover:text-ink-900'
                )
              }
            >
              <item.icon className="h-4 w-4 flex-none" strokeWidth={2} />
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-mist-200 p-4">
          {isAuthenticated && user ? (
            <button
              onClick={() => setProfileModalOpen(true)}
              className="flex w-full items-center gap-3 rounded-xl bg-mist-100 p-3 text-left transition-colors hover:bg-mist-200/80"
            >
              <div className="flex h-9 w-9 flex-none items-center justify-center rounded-full bg-trust-500 text-xs font-bold text-white shadow-xs">
                {initials}
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-semibold text-ink-900">{user.fullName}</p>
                <p className="truncate text-xs text-ink-500 capitalize">{user.role}</p>
              </div>
            </button>
          ) : (
            <button
              onClick={() => setAuthModalOpen(true)}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-mist-300 bg-white py-2.5 text-xs font-semibold text-ink-900 shadow-xs hover:bg-mist-50"
            >
              <LogIn className="h-4 w-4 text-ink-500" />
              Sign in / Register
            </button>
          )}
        </div>
      </aside>

      <div className="flex min-h-screen flex-1 flex-col lg:pl-64">
        <header className="sticky top-0 z-20 flex items-center justify-between gap-4 border-b border-mist-200 bg-white/90 px-6 py-3.5 backdrop-blur-md">
          <div className="flex max-w-md flex-1 items-center gap-2 rounded-xl border border-mist-200 bg-mist-50 px-3 py-2">
            <Search className="h-4 w-4 text-ink-400" />
            <input
              placeholder="Search claims, evidence, DOIs…"
              className="w-full bg-transparent text-sm text-ink-800 placeholder:text-ink-400 focus:outline-none"
            />
          </div>
          <div className="flex items-center gap-3">
            <Gauge className="h-4 w-4 text-verdict-support" />
            <span className="hidden text-xs text-ink-500 sm:inline">PostgreSQL Sync Active</span>
            {isAuthenticated && user ? (
              <button
                onClick={() => setProfileModalOpen(true)}
                className="flex items-center gap-2 rounded-lg bg-mist-100 px-3 py-1.5 text-xs font-semibold text-ink-900 hover:bg-mist-200"
              >
                <User className="h-3.5 w-3.5 text-trust-500" />
                {user.fullName.split(' ')[0]}
              </button>
            ) : (
              <button
                onClick={() => setAuthModalOpen(true)}
                className="rounded-lg bg-ink-950 px-3 py-1.5 text-xs font-semibold text-white hover:bg-ink-900"
              >
                Sign in
              </button>
            )}
          </div>
        </header>
        <main className="flex-1 px-6 py-8">
          <div className="mx-auto max-w-7xl">
            <Outlet />
          </div>
        </main>
      </div>

      {/* Global Auth Modals */}
      <AuthModal />
      <UserProfileModal />
    </div>
  )
}
