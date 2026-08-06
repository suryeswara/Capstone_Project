import { ShieldCheck } from 'lucide-react'

export function Footer() {
  return (
    <footer className="border-t border-mist-200 bg-white">
      <div className="mx-auto max-w-7xl px-6 py-14">
        <div className="grid grid-cols-2 gap-10 md:grid-cols-5">
          <div className="col-span-2">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-md bg-ink-950">
                <ShieldCheck className="h-4 w-4 text-clinical-400" />
              </div>
              <span className="font-semibold text-ink-950">MedVerify AI</span>
            </div>
            <p className="mt-3 max-w-xs text-sm leading-relaxed text-ink-500">
              Evidence-grounded medical claim verification with mandatory faithfulness checks
              on every AI explanation. Research-grade, not a chatbot.
            </p>
            <p className="mt-6 text-xs text-ink-400">
              Not a substitute for professional medical advice. Always consult a qualified
              clinician for individual health decisions.
            </p>
          </div>
          <div>
            <p className="mono-label mb-3">Product</p>
            <ul className="space-y-2 text-sm text-ink-600">
              <li><a href="#pipeline" className="hover:text-ink-950">Pipeline</a></li>
              <li><a href="#evidence" className="hover:text-ink-950">Evidence sources</a></li>
              <li><a href="#research" className="hover:text-ink-950">Research</a></li>
            </ul>
          </div>
          <div>
            <p className="mono-label mb-3">Platform</p>
            <ul className="space-y-2 text-sm text-ink-600">
              <li><a href="/app/dashboard" className="hover:text-ink-950">Dashboard</a></li>
              <li><a href="/app/history" className="hover:text-ink-950">History</a></li>
              <li><a href="/app/analytics" className="hover:text-ink-950">Analytics</a></li>
            </ul>
          </div>
          <div>
            <p className="mono-label mb-3">Trust</p>
            <ul className="space-y-2 text-sm text-ink-600">
              <li><a href="#faq" className="hover:text-ink-950">Methodology</a></li>
              <li><a href="#faq" className="hover:text-ink-950">FAQ</a></li>
              <li><a href="#" className="hover:text-ink-950">Security &amp; privacy</a></li>
            </ul>
          </div>
        </div>
        <div className="mt-12 flex flex-col gap-2 border-t border-mist-200 pt-6 text-xs text-ink-400 sm:flex-row sm:items-center sm:justify-between">
          <span>© 2026 MedVerify AI Research Project. Built for research demonstration purposes.</span>
          <span>Phase 1 MVP · 3 disease categories · English</span>
        </div>
      </div>
    </footer>
  )
}
