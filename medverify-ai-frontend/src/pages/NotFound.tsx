import { Link } from 'react-router-dom'
import { ShieldCheck } from 'lucide-react'
import { Button } from '@/components/ui/Primitives'

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-mist-50 px-6 text-center">
      <ShieldCheck className="h-10 w-10 text-clinical-500" />
      <h1 className="text-2xl font-semibold text-ink-950">Page not found</h1>
      <p className="max-w-sm text-sm text-ink-500">
        The page you&rsquo;re looking for doesn&rsquo;t exist, or the claim it references may have been removed.
      </p>
      <Link to="/"><Button>Back to home</Button></Link>
    </div>
  )
}
