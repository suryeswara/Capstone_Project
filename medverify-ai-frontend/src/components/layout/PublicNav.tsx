import { useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ShieldCheck, User, LogOut } from 'lucide-react'
import { Button } from '@/components/ui/Primitives'
import { useAppStore } from '@/store/useAppStore'
import { AuthModal } from '@/components/auth/AuthModal'
import { UserProfileModal } from '@/components/auth/UserProfileModal'

export function PublicNav() {
  const navigate = useNavigate()
  const { isAuthenticated, user, setAuthModalOpen, setProfileModalOpen, loadStoredAuth } = useAppStore()

  useEffect(() => {
    loadStoredAuth()
  }, [])

  const userInitials = user?.fullName
    ? user.fullName
        .split(' ')
        .map((n) => n[0])
        .join('')
        .toUpperCase()
    : 'U'

  return (
    <>
      <header className="sticky top-0 z-40 border-b border-mist-200/70 bg-mist-50/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink-950">
              <ShieldCheck className="h-4.5 w-4.5 text-clinical-400" strokeWidth={2.25} />
            </div>
            <span className="text-[15px] font-semibold tracking-tight text-ink-950">MedVerify AI</span>
          </Link>
          <nav className="hidden items-center gap-8 text-sm font-medium text-ink-600 md:flex">
            <a href="#pipeline" className="hover:text-ink-950">How it works</a>
            <a href="#evidence" className="hover:text-ink-950">Evidence system</a>
            <a href="#research" className="hover:text-ink-950">Research</a>
            <a href="#faq" className="hover:text-ink-950">FAQ</a>
          </nav>
          <div className="flex items-center gap-3">
            {isAuthenticated && user ? (
              <button
                onClick={() => setProfileModalOpen(true)}
                className="flex items-center gap-2.5 rounded-xl border border-mist-200 bg-white p-1.5 pr-3 shadow-xs hover:border-ink-300"
              >
                <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-trust-500 text-xs font-bold text-white">
                  {userInitials}
                </div>
                <span className="text-xs font-medium text-ink-900">{user.fullName.split(' ')[0]}</span>
              </button>
            ) : (
              <Button variant="ghost" className="hidden sm:inline-flex" onClick={() => setAuthModalOpen(true)}>
                Sign in
              </Button>
            )}
            <Button onClick={() => navigate('/app/verify')}>Verify a claim</Button>
          </div>
        </div>
      </header>

      {/* Global Modals */}
      <AuthModal />
      <UserProfileModal />
    </>
  )
}
