import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

export function CredibilityGauge({
  score,
  size = 168,
  label = 'Overall credibility',
  toneOverride,
}: {
  score: number
  size?: number
  label?: string
  toneOverride?: string
}) {
  const stroke = size * 0.09
  const r = (size - stroke) / 2
  const circumference = 2 * Math.PI * r
  const tone =
    toneOverride ??
    (score >= 85 ? '#1E9E6B' : score >= 60 ? '#C88A1E' : '#C4453B')

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={r} stroke="#EEF2F6" strokeWidth={stroke} fill="none" />
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            stroke={tone}
            strokeWidth={stroke}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: circumference - (score / 100) * circumference }}
            transition={{ duration: 1.1, ease: 'easeOut', delay: 0.15 }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="text-3xl font-semibold tabular-nums text-ink-950"
          >
            {score}
          </motion.span>
          <span className="mono-label mt-0.5">/ 100</span>
        </div>
      </div>
      <p className={cn('mt-3 text-center text-sm font-medium text-ink-700')}>{label}</p>
    </div>
  )
}

export function MiniMeter({ label, value }: { label: string; value: number }) {
  const tone = value >= 85 ? 'bg-verdict-support' : value >= 60 ? 'bg-verdict-warn' : 'bg-verdict-contradict'
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-sm">
        <span className="text-ink-600">{label}</span>
        <span className="font-medium tabular-nums text-ink-900">{value}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-mist-150 bg-mist-100">
        <motion.div
          className={cn('h-full rounded-full', tone)}
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.9, ease: 'easeOut' }}
        />
      </div>
    </div>
  )
}
