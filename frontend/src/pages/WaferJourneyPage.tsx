import { useEffect } from 'react'
import { useAppStore } from '../lib/store'
import { badBatch } from '../lib/fixtures'
import WaferJourney from '../components/WaferJourney'
import ShapPanel from '../components/ShapPanel'
import ErrorBanner from '../components/ErrorBanner'

export default function WaferJourneyPage() {
  const { activeBatch, setActiveBatch, selectedStep } = useAppStore()

  useEffect(() => {
    if (!activeBatch) setActiveBatch(badBatch)
  }, [activeBatch, setActiveBatch])

  const batch = activeBatch ?? badBatch
  const anomalyCount = batch.per_step.filter((s) => s.risk_score >= 0.70).length
  const isHealthy = anomalyCount === 0

  return (
    <div className="space-y-4">
      <ErrorBanner />

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-white font-bold text-xl">Wafer Inspector</h1>
          <p className="text-white/40 text-xs font-mono mt-0.5">
            {batch.batch_id} · {batch.per_step.length} fabrication steps · Quality Score:{' '}
            <span className={batch.predicted_yield >= 0.85 ? 'text-yield-green' : batch.predicted_yield >= 0.70 ? 'text-risk-amber' : 'text-risk-red'}>
              {(batch.predicted_yield * 100).toFixed(1)}%
            </span>
          </p>
        </div>
        <div
          className="text-xs font-mono px-3 py-1.5 rounded-full font-bold"
          style={{
            backgroundColor: isHealthy ? '#00cc6620' : '#ff444420',
            color: isHealthy ? '#00cc66' : '#ff4444',
            border: `1px solid ${isHealthy ? '#00cc6640' : '#ff444440'}`,
          }}
        >
          {isHealthy ? '✅ All Clear' : `🚨 ${anomalyCount} Issues`}
        </div>
      </div>

      {/* Main layout: timeline left, SHAP panel right */}
      <div className="flex gap-4 items-start">
        {/* Timeline card — takes most of the width */}
        <div className="flex-1 min-w-0 bg-[#0d1426] rounded-xl p-4 border border-white/10 space-y-3">
          <div className="text-white/40 text-xs font-mono uppercase tracking-wider">
            Process Timeline — hover any step to inspect
          </div>
          <WaferJourney steps={batch.per_step} />
          {/* Legend */}
          <div className="flex flex-wrap gap-4 text-xs font-mono text-white/40 pt-1 border-t border-white/5">
            <span><span className="inline-block w-3 h-3 rounded-full bg-[#ff4444] mr-1.5 align-middle" />High Risk (≥85%)</span>
            <span><span className="inline-block w-3 h-3 rounded-full bg-[#ffaa00] mr-1.5 align-middle" />Medium Risk (70–84%)</span>
            <span><span className="inline-block w-3 h-3 rounded-full bg-[#1e2a3a] border border-white/20 mr-1.5 align-middle" />Normal</span>
            <span className="ml-auto text-white/20">Outer ring = process category</span>
          </div>
        </div>

        {/* SHAP side panel — fixed width, appears when a step is selected */}
        <div
          className="w-80 flex-shrink-0 bg-[#0d1426] rounded-xl p-4 border border-white/10 transition-all duration-300"
          style={{ opacity: selectedStep ? 1 : 0.4 }}
        >
          {selectedStep ? (
            <>
              <div className="text-white/40 text-xs font-mono uppercase tracking-wider mb-3">
                AI Explanation
              </div>
              <div className="mb-1 font-mono text-sm font-bold text-white truncate">
                {selectedStep.step_name}
              </div>
              <div className="text-xs font-mono mb-3 flex items-center gap-2">
                <span
                  className="px-2 py-0.5 rounded-full text-xs font-bold"
                  style={{
                    backgroundColor: selectedStep.risk_score >= 0.85 ? '#ff444420' : selectedStep.risk_score >= 0.70 ? '#ffaa0020' : '#1e2a3a',
                    color: selectedStep.risk_score >= 0.85 ? '#ff4444' : selectedStep.risk_score >= 0.70 ? '#ffaa00' : '#ffffff60',
                  }}
                >
                  {(selectedStep.risk_score * 100).toFixed(0)}% failure risk
                </span>
                <span className="text-white/30">
                  CI: [{(selectedStep.confidence_lo * 100).toFixed(0)}–{(selectedStep.confidence_hi * 100).toFixed(0)}%]
                </span>
              </div>
              <ShapPanel shap={selectedStep.shap} />
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-48 text-center gap-2">
              <div className="text-2xl">👆</div>
              <div className="text-white/30 text-xs font-mono">
                Hover a step in the timeline<br />to see AI explanations
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Risk step list — only show if anomalies exist */}
      {anomalyCount > 0 && (
        <div className="bg-[#0d1426] rounded-xl p-4 border border-white/10">
          <div className="text-white/40 text-xs font-mono uppercase tracking-wider mb-3">
            Flagged Steps ({anomalyCount})
          </div>
          <div className="grid grid-cols-2 gap-2">
            {batch.per_step
              .filter((s) => s.risk_score >= 0.70)
              .slice(0, 12)
              .map((s) => (
                <div
                  key={s.step_id}
                  className="flex items-center justify-between rounded-lg px-3 py-2 border"
                  style={{
                    backgroundColor: s.risk_score >= 0.85 ? '#ff444410' : '#ffaa0010',
                    borderColor: s.risk_score >= 0.85 ? '#ff444430' : '#ffaa0030',
                  }}
                >
                  <span className="text-white/70 text-xs font-mono truncate max-w-[160px]">{s.step_name}</span>
                  <span
                    className="text-xs font-mono font-bold ml-2 flex-shrink-0"
                    style={{ color: s.risk_score >= 0.85 ? '#ff4444' : '#ffaa00' }}
                  >
                    {(s.risk_score * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
          </div>
          {anomalyCount > 12 && (
            <div className="text-white/20 text-xs font-mono mt-2">+ {anomalyCount - 12} more flagged steps</div>
          )}
        </div>
      )}
    </div>
  )
}
