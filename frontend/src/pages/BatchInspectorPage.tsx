import BatchInspector from '../components/BatchInspector'
import ErrorBanner from '../components/ErrorBanner'

export default function BatchInspectorPage() {
  return (
    <div className="space-y-4">
      <ErrorBanner />
      <div>
        <h1 className="text-white font-mono text-lg">Batch Inspector</h1>
        <p className="text-white/40 text-xs font-mono mt-0.5">
          Click a row to inspect per-step risk breakdown
        </p>
      </div>
      <BatchInspector />
    </div>
  )
}
