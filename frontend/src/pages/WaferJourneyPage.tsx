import { useEffect } from 'react'
import { useAppStore } from '../lib/store'
import { badBatch } from '../lib/fixtures'
import WaferJourney from '../components/WaferJourney'
import ErrorBanner from '../components/ErrorBanner'

export default function WaferJourneyPage() {
  const { activeBatch, setActiveBatch } = useAppStore()

  useEffect(() => {
    if (!activeBatch) {
      setActiveBatch(badBatch)
    }
  }, [activeBatch, setActiveBatch])

  const batch = activeBatch ?? badBatch

  return (
    <div className="space-y-4">
      <ErrorBanner />
      <div>
        <h1 className="text-white font-mono text-lg mb-1">Wafer Journey</h1>
        <p className="text-white/40 text-xs font-mono">
          Batch: {batch.batch_id} — {batch.per_step.length} steps — Yield: {(batch.predicted_yield * 100).toFixed(1)}%
        </p>
      </div>
      <div className="bg-[#0d1426] rounded-lg p-4 border border-white/10">
        <WaferJourney steps={batch.per_step} />
      </div>
      <div className="flex gap-4 text-xs font-mono text-white/50">
        <span><span className="inline-block w-3 h-3 rounded-full bg-[#ff4444] mr-1" />High Risk (≥0.85)</span>
        <span><span className="inline-block w-3 h-3 rounded-full bg-[#ffaa00] mr-1" />Medium Risk (0.70–0.84)</span>
        <span><span className="inline-block w-3 h-3 rounded-full bg-[#1e2a3a] border border-white/20 mr-1" />Normal</span>
      </div>
    </div>
  )
}
