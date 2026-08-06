import { clsx, type ClassValue } from 'clsx'

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs)
}

export function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

export function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  return `${days}d ago`
}

export function verdictTone(verdict: string) {
  switch (verdict) {
    case 'Supported':
      return { text: 'text-verdict-support', bg: 'bg-verdict-support/10', ring: 'ring-verdict-support/30', dot: 'bg-verdict-support' }
    case 'Contradicted':
      return { text: 'text-verdict-contradict', bg: 'bg-verdict-contradict/10', ring: 'ring-verdict-contradict/30', dot: 'bg-verdict-contradict' }
    case 'Mixed':
      return { text: 'text-verdict-warn', bg: 'bg-verdict-warn/10', ring: 'ring-verdict-warn/30', dot: 'bg-verdict-warn' }
    default:
      return { text: 'text-verdict-neutral', bg: 'bg-verdict-neutral/10', ring: 'ring-verdict-neutral/30', dot: 'bg-verdict-neutral' }
  }
}
