import { motion, AnimatePresence } from 'framer-motion'
import { Check, Loader2, AlertTriangle, Circle } from 'lucide-react'
import type { PipelineStage } from '@/types'
import { cn } from '@/lib/utils'

function StageIcon({ status }: { status: PipelineStage['status'] }) {
  if (status === 'complete') return <Check className="h-4 w-4" strokeWidth={2.5} />
  if (status === 'active') return <Loader2 className="h-4 w-4 animate-spin" />
  if (status === 'flagged') return <AlertTriangle className="h-4 w-4" />
  return <Circle className="h-3 w-3" />
}

export function PipelineVisualizer({ stages, compact = false }: { stages: PipelineStage[]; compact?: boolean }) {
  return (
    <div className="relative">
      {/* the evidence thread */}
      <div className="absolute left-[15px] top-2 bottom-2 w-px bg-mist-200" aria-hidden />
      <ol className="relative space-y-0">
        {stages.map((stage, i) => (
          <li key={stage.id} className="relative flex gap-4 pb-6 last:pb-0">
            <div className="relative z-10 flex h-8 w-8 flex-none items-center justify-center">
              <motion.div
                className={cn(
                  'flex h-8 w-8 items-center justify-center rounded-full border-2 transition-colors duration-300',
                  stage.status === 'complete' && 'border-clinical-500 bg-clinical-500 text-white',
                  stage.status === 'active' && 'border-clinical-500 bg-white text-clinical-600 animate-pulseRing',
                  stage.status === 'pending' && 'border-mist-300 bg-white text-mist-400',
                  stage.status === 'flagged' && 'border-verdict-warn bg-white text-verdict-warn'
                )}
              >
                <StageIcon status={stage.status} />
              </motion.div>
              {stage.status === 'complete' && i < stages.length - 1 && (
                <motion.div
                  className="absolute left-1/2 top-8 h-6 w-px -translate-x-1/2 bg-clinical-500"
                  initial={{ scaleY: 0 }}
                  animate={{ scaleY: 1 }}
                  style={{ transformOrigin: 'top' }}
                />
              )}
            </div>
            <div className="flex-1 pt-0.5">
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    'text-sm font-medium',
                    stage.status === 'pending' ? 'text-ink-400' : 'text-ink-900'
                  )}
                >
                  {stage.label}
                </span>
                <AnimatePresence>
                  {stage.status === 'active' && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="mono-label text-clinical-600"
                    >
                      processing
                    </motion.span>
                  )}
                  {stage.status === 'complete' && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="mono-label text-verdict-support"
                    >
                      done
                    </motion.span>
                  )}
                </AnimatePresence>
              </div>
              {!compact && (
                <p className={cn('mt-1 text-sm text-ink-500', stage.status === 'pending' && 'text-ink-400/70')}>
                  {stage.description}
                </p>
              )}
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
