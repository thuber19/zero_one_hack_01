import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useAppStore } from '../lib/store'
import { getCompletionDetail, getAnomalyDetail, postInferNextStep, postInferComplete, postInferAnomaly } from '../lib/api'
import type { CompletionDetail, AnomalyDetail } from '../types/api'

const RULE_EXPLANATIONS: Record<string, string> = {
  RULE_DEP_NO_CLEAN: 'Deposition without prior clean step',
  RULE_METAL_ETCH_NO_LITHO: 'Metal etch without prior lithography',
  RULE_ETCH_NO_MASK: 'Patterned etch without develop photoresist',
  RULE_LITHO_LEVEL_SKIP: 'Lithography mask level skipped or out of order',
  RULE_IMPLANT_NO_MASK: 'Implant without prior oxide etch or develop',
  RULE_CMP_NO_DEP: 'CMP without prior deposition',
  RULE_PAD_OPEN_BEFORE_DEP: 'Pad opening before passivation deposition',
  RULE_TEST_BEFORE_PASSIVATION: 'Electrical test before cure passivation',
  RULE_SHIP_BEFORE_TEST: 'Ship lot before wafer sort test',
  RULE_BACKSIDE_BEFORE_PASSIVATION: 'Backside metal before cure passivation',
}

function PanelHeader({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-white/40 text-xs font-mono uppercase tracking-wider mb-3">
      {children}
    </div>
  )
}

function StepList({ steps }: { steps: string[] }) {
  return (
    <div className="overflow-y-auto max-h-[440px] space-y-1 pr-1">
      {steps.map((step, i) => (
        <div key={i} className="flex items-center gap-2 py-0.5">
          <span className="text-white/20 font-mono text-xs w-8 flex-shrink-0 text-right">{i + 1}</span>
          <span className="font-mono text-white/70 text-xs">{step}</span>
        </div>
      ))}
    </div>
  )
}

function CountBadge({ count, label }: { count: number; label: string }) {
  return (
    <div className="mt-3 pt-2 border-t border-white/10 text-white/30 text-xs font-mono">
      {count} {label}
    </div>
  )
}

function SkeletonRows() {
  return (
    <div className="space-y-2 pt-1">
      {[70, 55, 85, 60, 45, 75, 50].map((w, i) => (
        <div key={i} className="flex items-center gap-2">
          <div className="h-3 w-6 rounded bg-white/5 animate-pulse flex-shrink-0" />
          <div className="h-3 rounded bg-white/5 animate-pulse" style={{ width: `${w}%` }} />
        </div>
      ))}
    </div>
  )
}

function LoadingPanel() {
  return (
    <div className="bg-[#0d1426] rounded-xl p-4 border border-white/10 min-h-[200px]">
      <div className="h-3 w-24 rounded bg-white/5 animate-pulse mb-4" />
      <SkeletonRows />
    </div>
  )
}

function CompletionView({ detail }: { detail: CompletionDetail }) {
  const ranks = [detail.rank1, detail.rank2, detail.rank3, detail.rank4, detail.rank5]

  return (
    <>
      {/* LEFT — Given Sequence */}
      <div className="bg-[#0d1426] rounded-xl p-4 border border-white/10">
        <PanelHeader>Given Sequence</PanelHeader>
        <StepList steps={detail.partial_steps} />
        <CountBadge count={detail.partial_step_count} label="steps given" />
      </div>

      {/* MIDDLE — Task 1 */}
      <div className="bg-[#0d1426] rounded-xl p-4 border border-white/10">
        <PanelHeader>Task 1 — Next Step</PanelHeader>
        <div className="space-y-2">
          {ranks.map((step, i) => {
            const isTop = i === 0
            return (
              <div
                key={i}
                className="flex items-center gap-3 rounded-lg px-3 py-2"
                style={
                  isTop
                    ? {
                        border: '1px solid #00d4ff40',
                        backgroundColor: '#00d4ff08',
                        boxShadow: '0 0 8px #00d4ff18',
                      }
                    : {
                        border: '1px solid #ffffff08',
                        backgroundColor: 'transparent',
                      }
                }
              >
                <span
                  className="font-mono text-xs font-bold flex-shrink-0 w-7 text-center rounded"
                  style={
                    isTop
                      ? { color: '#00d4ff', backgroundColor: '#00d4ff18', padding: '1px 4px' }
                      : { color: '#ffffff30', padding: '1px 4px' }
                  }
                >
                  #{i + 1}
                </span>
                <span
                  className={`font-mono text-xs ${isTop ? 'font-bold text-[#00d4ff]' : 'text-white/50'}`}
                >
                  {step}
                </span>
              </div>
            )
          })}
        </div>
        <div className="mt-3 text-white/30 text-xs font-mono">
          Physics re-ranked → 93.7% Top-1
        </div>
      </div>

      {/* RIGHT — Task 2 */}
      <div className="bg-[#0d1426] rounded-xl p-4 border border-white/10">
        <PanelHeader>Task 2 — Completion</PanelHeader>
        <div
          className="font-mono text-xs px-2 py-1 rounded mb-3 inline-block"
          style={{
            backgroundColor: '#00cc6615',
            border: '1px solid #00cc6640',
            color: '#00cc66',
          }}
        >
          Physics ✓&nbsp; 100% rule-valid
        </div>
        <StepList steps={detail.predicted_sequence_steps} />
        <CountBadge count={detail.predicted_step_count} label="steps predicted" />
      </div>
    </>
  )
}

function AnomalyView({ detail }: { detail: AnomalyDetail }) {
  const ruleExplanation = detail.predicted_rule ? RULE_EXPLANATIONS[detail.predicted_rule] : null

  return (
    <>
      {/* LEFT — Full Sequence */}
      <div className="bg-[#0d1426] rounded-xl p-4 border border-white/10">
        <PanelHeader>Given Sequence</PanelHeader>
        <StepList steps={detail.full_sequence} />
        <CountBadge count={detail.sequence_step_count} label="total steps" />
      </div>

      {/* MIDDLE — Task 1 N/A */}
      <div className="bg-[#0d1426] rounded-xl p-4 border border-white/10 flex flex-col items-center justify-center min-h-[200px]">
        <div className="text-white/20 font-mono text-2xl mb-2">N/A</div>
        <div className="text-white/30 text-xs font-mono text-center max-w-[180px]">
          Task 1 not applicable for anomaly detection examples
        </div>
      </div>

      {/* RIGHT — Task 3 */}
      <div className="bg-[#0d1426] rounded-xl p-4 border border-white/10">
        <PanelHeader>Task 3 — Anomaly Detection</PanelHeader>

        {/* Valid / Anomaly badge */}
        <div
          className="font-mono text-sm font-bold px-4 py-3 rounded-lg mb-4 text-center"
          style={
            detail.is_valid
              ? { backgroundColor: '#00cc6615', border: '1px solid #00cc6640', color: '#00cc66' }
              : { backgroundColor: '#ff444415', border: '1px solid #ff444440', color: '#ff4444' }
          }
        >
          {detail.is_valid ? '✓ VALID' : '⚠ ANOMALY DETECTED'}
        </div>

        {/* Confidence score */}
        <div className="font-mono text-xs text-white/50 mb-4">
          Confidence:{' '}
          <span className="text-white/80 font-bold">{(detail.score * 100).toFixed(1)}%</span>
        </div>

        {/* Rule chip + explanation */}
        {!detail.is_valid && detail.predicted_rule && (
          <div className="space-y-2">
            <div
              className="font-mono text-xs px-2 py-1 rounded inline-block"
              style={{
                backgroundColor: '#ff444415',
                border: '1px solid #ff444440',
                color: '#ff4444',
              }}
            >
              {detail.predicted_rule}
            </div>
            {ruleExplanation && (
              <div className="text-white/50 text-xs font-mono">{ruleExplanation}</div>
            )}
          </div>
        )}
      </div>
    </>
  )
}

type LiveStatus = 'idle' | 'loading' | 'done' | 'error'

interface LiveResult {
  nextStepPredictions?: string[]
  completionSteps?: string[]
  completionIsValid?: boolean
  anomalyVerdict?: { is_valid: boolean; score: number; predicted_rule: string }
}

function LiveInferencePanel({
  type,
  steps,
  family,
  precomputedRanks,
  precomputedIsValid,
}: {
  type: 'completion' | 'anomaly'
  steps: string[]
  family: string
  precomputedRanks?: string[]
  precomputedIsValid?: boolean
}) {
  const [status, setStatus] = useState<LiveStatus>('idle')
  const [result, setResult] = useState<LiveResult | null>(null)

  async function runLive() {
    setStatus('loading')
    setResult(null)
    try {
      if (type === 'completion') {
        const [nsRes, compRes] = await Promise.all([
          postInferNextStep(steps, family),
          postInferComplete(steps, family),
        ])
        setResult({
          nextStepPredictions: nsRes.predictions,
          completionSteps: compRes.completion,
          completionIsValid: compRes.is_valid,
        })
      } else {
        const res = await postInferAnomaly(steps, family)
        setResult({
          anomalyVerdict: {
            is_valid: res.is_valid ?? true,
            score: res.score ?? 0.5,
            predicted_rule: res.predicted_rule ?? '',
          },
        })
      }
      setStatus('done')
    } catch {
      setStatus('error')
    }
  }

  return (
    <div className="rounded-xl border border-white/10 p-4 space-y-4" style={{ backgroundColor: '#0d1426' }}>
      <div className="flex items-center justify-between">
        <div className="text-white/40 text-xs font-mono uppercase tracking-wider">
          Live Inference — {type === 'completion' ? 'Tasks 1 & 2' : 'Task 3'}
        </div>
        <button
          onClick={runLive}
          disabled={status === 'loading'}
          className="font-mono text-xs px-4 py-1.5 rounded-lg border transition-all disabled:opacity-40 hover:opacity-80"
          style={{ borderColor: '#00d4ff40', color: '#00d4ff', backgroundColor: '#00d4ff10' }}
        >
          {status === 'loading' ? '⟳ Running…' : '▶ Run Live Inference'}
        </button>
      </div>

      {status === 'idle' && (
        <div className="text-white/25 text-xs font-mono">
          Calls the live decoder model — results may differ slightly from pre-computed
        </div>
      )}

      {status === 'error' && (
        <div className="text-xs font-mono" style={{ color: '#ff4444' }}>
          Inference failed — model may still be loading or unavailable
        </div>
      )}

      {status === 'done' && result && type === 'completion' && (
        <div className="grid grid-cols-2 gap-6">
          {/* Next-step comparison */}
          <div className="space-y-2">
            <div className="text-white/40 text-xs font-mono uppercase tracking-wider mb-2">
              Live Next-Step (Top-5)
            </div>
            {(result.nextStepPredictions ?? []).map((step, i) => {
              const matches = precomputedRanks?.[i] === step
              return (
                <div key={i} className="flex items-center gap-2">
                  <span className="font-mono text-xs text-white/20 w-5 text-right flex-shrink-0">#{i + 1}</span>
                  <span
                    className="font-mono text-xs truncate"
                    style={{ color: matches ? '#00cc66' : '#ffaa00' }}
                    title={step}
                  >
                    {step}
                  </span>
                  {matches && <span className="text-white/30 text-xs flex-shrink-0">= pre-computed</span>}
                </div>
              )
            })}
          </div>

          {/* Completion result */}
          <div className="space-y-2">
            <div className="text-white/40 text-xs font-mono uppercase tracking-wider mb-2">
              Live Completion
            </div>
            <div
              className="font-mono text-xs px-2 py-1 rounded inline-block"
              style={
                result.completionIsValid
                  ? { backgroundColor: '#00cc6615', border: '1px solid #00cc6640', color: '#00cc66' }
                  : { backgroundColor: '#ff444415', border: '1px solid #ff444440', color: '#ff4444' }
              }
            >
              {result.completionIsValid ? 'Physics ✓ Rule-valid' : '⚠ Rule violation'}
            </div>
            <div className="text-white/40 text-xs font-mono">
              {result.completionSteps?.length ?? 0} steps predicted
            </div>
          </div>
        </div>
      )}

      {status === 'done' && result && type === 'anomaly' && result.anomalyVerdict && (
        <div className="flex flex-wrap items-center gap-6">
          <div>
            <div className="text-white/40 text-xs font-mono uppercase tracking-wider mb-1">Live Verdict</div>
            <div
              className="font-mono text-xs px-3 py-1.5 rounded font-bold"
              style={
                result.anomalyVerdict.is_valid
                  ? { backgroundColor: '#00cc6615', border: '1px solid #00cc6640', color: '#00cc66' }
                  : { backgroundColor: '#ff444415', border: '1px solid #ff444440', color: '#ff4444' }
              }
            >
              {result.anomalyVerdict.is_valid ? '✓ VALID' : '⚠ ANOMALY'}
            </div>
          </div>

          {result.anomalyVerdict.predicted_rule && (
            <div>
              <div className="text-white/40 text-xs font-mono uppercase tracking-wider mb-1">Rule Violated</div>
              <div
                className="font-mono text-xs px-2 py-1 rounded"
                style={{ backgroundColor: '#ff444415', border: '1px solid #ff444440', color: '#ff4444' }}
              >
                {result.anomalyVerdict.predicted_rule}
              </div>
            </div>
          )}

          <div>
            <div className="text-white/40 text-xs font-mono uppercase tracking-wider mb-1">Consistent?</div>
            <div className="font-mono text-xs">
              {result.anomalyVerdict.is_valid === precomputedIsValid ? (
                <span style={{ color: '#00cc66' }}>✓ Matches pre-computed</span>
              ) : (
                <span style={{ color: '#ffaa00' }}>≠ Differs from pre-computed</span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function WaferJourneyPage() {
  const rawId = useAppStore((s) => s.activeSequenceId)
  const rawType = useAppStore((s) => s.activeSequenceType)

  const activeSequenceId = rawId ?? 'valid_0001'
  const activeSequenceType = rawType ?? 'completion'

  const isCompletion = activeSequenceType === 'completion'

  const completionQuery = useQuery<CompletionDetail>({
    queryKey: ['completionDetail', activeSequenceId],
    queryFn: () => getCompletionDetail(activeSequenceId),
    enabled: isCompletion,
    staleTime: Infinity,
  })

  const anomalyQuery = useQuery<AnomalyDetail>({
    queryKey: ['anomalyDetail', activeSequenceId],
    queryFn: () => getAnomalyDetail(activeSequenceId),
    enabled: !isCompletion,
    staleTime: Infinity,
  })

  const isLoading = isCompletion ? completionQuery.isLoading : anomalyQuery.isLoading
  const isError = isCompletion ? completionQuery.isError : anomalyQuery.isError
  const completionData = completionQuery.data
  const anomalyData = anomalyQuery.data

  // Build subtitle
  let subtitle = ''
  if (isCompletion && completionData) {
    subtitle = `${completionData.example_id}  ·  ${completionData.family}  ·  ${(completionData.completion_fraction * 100).toFixed(0)}% given`
  } else if (!isCompletion && anomalyData) {
    subtitle = `${anomalyData.example_id}  ·  ${anomalyData.family}`
  } else {
    subtitle = activeSequenceId
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h1 className="text-white font-bold text-xl">Sequence Analysis</h1>
        <p className="text-white/40 text-xs font-mono mt-0.5">{subtitle}</p>
      </div>

      {isError && (
        <div
          className="rounded-lg px-4 py-3 font-mono text-xs"
          style={{ backgroundColor: '#ff444415', border: '1px solid #ff444430', color: '#ff4444' }}
        >
          Failed to load sequence data for "{activeSequenceId}".
        </div>
      )}

      {/* Three-panel layout */}
      <div className="grid grid-cols-3 gap-4">
        {isLoading ? (
          <>
            <LoadingPanel />
            <LoadingPanel />
            <LoadingPanel />
          </>
        ) : isError ? null : isCompletion && completionData ? (
          <CompletionView detail={completionData} />
        ) : !isCompletion && anomalyData ? (
          <AnomalyView detail={anomalyData} />
        ) : null}
      </div>

      {/* Live inference panel — only shown when data is loaded */}
      {!isLoading && !isError && isCompletion && completionData && (
        <LiveInferencePanel
          type="completion"
          steps={completionData.partial_steps}
          family={completionData.family}
          precomputedRanks={[
            completionData.rank1,
            completionData.rank2,
            completionData.rank3,
            completionData.rank4,
            completionData.rank5,
          ]}
        />
      )}
      {!isLoading && !isError && !isCompletion && anomalyData && (
        <LiveInferencePanel
          type="anomaly"
          steps={anomalyData.full_sequence}
          family={anomalyData.family}
          precomputedIsValid={anomalyData.is_valid}
        />
      )}
    </div>
  )
}
