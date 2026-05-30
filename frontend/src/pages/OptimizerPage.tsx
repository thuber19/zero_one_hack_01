import { useAppStore } from '../lib/store'
import { badBatch } from '../lib/fixtures'
import SequenceOptimizer from '../components/SequenceOptimizer'
import ErrorBanner from '../components/ErrorBanner'

export default function OptimizerPage() {
  const activeBatch = useAppStore((s) => s.activeBatch) ?? badBatch
  const steps = activeBatch.per_step.slice(0, 20)

  return (
    <div className="space-y-4">
      <ErrorBanner />
      <h1 className="text-white font-mono text-lg">Sequence Optimizer</h1>
      <div className="bg-[#0d1426] rounded-lg p-4 border border-white/10">
        <SequenceOptimizer steps={steps} />
      </div>
    </div>
  )
}
