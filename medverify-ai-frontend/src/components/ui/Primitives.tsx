import type { ReactNode, ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('rounded-2xl border border-mist-200 bg-white shadow-card', className)}>
      {children}
    </div>
  )
}

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('mono-label', className)}>{children}</div>
}

export function Badge({
  children,
  tone = 'neutral',
  className,
}: {
  children: ReactNode
  tone?: 'neutral' | 'clinical' | 'trust' | 'support' | 'warn' | 'contradict'
  className?: string
}) {
  const tones: Record<string, string> = {
    neutral: 'bg-mist-100 text-ink-600 ring-mist-300',
    clinical: 'bg-clinical-50 text-clinical-700 ring-clinical-400/30',
    trust: 'bg-trust-500/10 text-trust-700 ring-trust-500/20',
    support: 'bg-verdict-support/10 text-verdict-support ring-verdict-support/25',
    warn: 'bg-verdict-warn/10 text-verdict-warn ring-verdict-warn/25',
    contradict: 'bg-verdict-contradict/10 text-verdict-contradict ring-verdict-contradict/25',
  }
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1',
        tones[tone],
        className
      )}
    >
      {children}
    </span>
  )
}

export function Button({
  children,
  variant = 'primary',
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'ghost' }) {
  const variants: Record<string, string> = {
    primary: 'bg-ink-900 text-white hover:bg-ink-800 active:bg-ink-950 shadow-sm',
    secondary: 'bg-white text-ink-800 border border-mist-300 hover:border-ink-500 hover:bg-mist-50',
    ghost: 'text-ink-600 hover:bg-mist-100',
  }
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-colors disabled:opacity-50 disabled:pointer-events-none',
        variants[variant],
        className
      )}
      {...props}
    >
      {children}
    </button>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('shimmer rounded-lg', className)} />
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  className,
}: {
  eyebrow?: string
  title: string
  description?: string
  className?: string
}) {
  return (
    <div className={cn('max-w-2xl', className)}>
      {eyebrow && <Eyebrow className="mb-3 text-clinical-600">{eyebrow}</Eyebrow>}
      <h2 className="text-3xl font-semibold tracking-tight text-ink-950 sm:text-4xl">{title}</h2>
      {description && <p className="mt-3 text-base leading-relaxed text-ink-600">{description}</p>}
    </div>
  )
}
