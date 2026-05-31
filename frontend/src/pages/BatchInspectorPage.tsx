import { useState } from 'react'
import { CompletionsTable, AnomaliesTable } from '../components/BatchInspector'
import ErrorBanner from '../components/ErrorBanner'

type Tab = 'completions' | 'anomalies'

export default function BatchInspectorPage() {
  const [tab, setTab] = useState<Tab>('completions')

  return (
    <div className="space-y-4">
      <ErrorBanner />
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-white font-bold text-xl">Eval Results Browser</h1>
          <p className="text-white/40 text-xs font-mono mt-0.5">
            600 sequence completions (Tasks 1 & 2) + 987 anomaly detections (Task 3) · click any row to inspect
          </p>
        </div>
      </div>

      {/* Tab switcher */}
      <div className="flex gap-2">
        <button
          onClick={() => setTab('completions')}
          className={`text-xs font-mono px-4 py-2 rounded-lg border transition-colors ${
            tab === 'completions'
              ? 'border-[#00d4ff40] bg-[#00d4ff15] text-[#00d4ff]'
              : 'border-white/10 text-white/40 hover:text-white hover:border-white/20'
          }`}
        >
          Completions (600)
        </button>
        <button
          onClick={() => setTab('anomalies')}
          className={`text-xs font-mono px-4 py-2 rounded-lg border transition-colors ${
            tab === 'anomalies'
              ? 'border-[#ff444440] bg-[#ff444415] text-[#ff4444]'
              : 'border-white/10 text-white/40 hover:text-white hover:border-white/20'
          }`}
        >
          Anomalies (987)
        </button>
      </div>

      {tab === 'completions' ? <CompletionsTable /> : <AnomaliesTable />}
    </div>
  )
}
