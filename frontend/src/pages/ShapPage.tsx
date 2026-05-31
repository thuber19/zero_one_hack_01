import { useAppStore } from '../lib/store'
import { badBatch } from '../lib/fixtures'
import WaferJourney from '../components/WaferJourney'
import ShapPanel from '../components/ShapPanel'
import ErrorBanner from '../components/ErrorBanner'

export default function ShapPage() {
  const { activeBatch, selectedStep } = useAppStore()
  const batch = activeBatch ?? badBatch

  return (
    <div className="space-y-4">
      <ErrorBanner />
      <h1 className="text-white font-mono text-lg">SHAP Explainability</h1>
      <p className="text-white/40 text-xs font-mono">
        Click any step node to see SHAP feature contributions
      </p>
      <div className="bg-[#0d1426] rounded-lg p-4 border border-white/10">
        <WaferJourney steps={batch.per_step} />
      </div>
      <div className="bg-[#0d1426] rounded-lg p-4 border border-white/10">
        <ShapPanel
          shap={selectedStep?.shap ?? null}
          stepName={selectedStep?.step_name}
        />
      </div>
    </div>
  )
}
