import BatchInspector from '../components/BatchInspector'
import ErrorBanner from '../components/ErrorBanner'

export default function BatchInspectorPage() {
  return (
    <div className="space-y-4">
      <ErrorBanner />
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-white font-bold text-xl">Batch Triage</h1>
          <p className="text-white/40 text-xs font-mono mt-0.5">
            1,587 real wafers scored by BERT MLM — sorted by failure risk · click any row to inspect
          </p>
        </div>
        <div className="flex gap-3 text-xs font-mono">
          <span className="flex items-center gap-1.5 text-white/40">
            <span className="inline-block w-2 h-2 rounded-sm bg-[#ff4444]" />High risk (F grade)
          </span>
          <span className="flex items-center gap-1.5 text-white/40">
            <span className="inline-block w-2 h-2 rounded-sm bg-[#ffaa00]" />Medium (C)
          </span>
          <span className="flex items-center gap-1.5 text-white/40">
            <span className="inline-block w-2 h-2 rounded-sm bg-[#00cc66]" />Healthy (A/B)
          </span>
        </div>
      </div>
      <BatchInspector />
    </div>
  )
}
