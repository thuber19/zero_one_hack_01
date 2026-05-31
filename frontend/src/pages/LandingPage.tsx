import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { postInferAnomaly } from '../lib/api'

// Short sequence that clearly triggers RULE_SHIP_BEFORE_TEST
// Chosen for readability — MOSFET_inval_0002 has the same violation over 126 steps
const DEMO_SEQUENCE = [
  'RECEIVE WAFER LOT',
  'LOT IDENTIFICATION',
  'PRE CLEAN INSPECTION',
  'THERMAL OXIDATION',
  'SPIN COAT PHOTORESIST',
  'EXPOSE LITHO LEVEL 1',
  'GATE OXIDE GROWTH',
  'DEPOSIT POLYSILICON',
  'PASSIVATION DEPOSITION',
  'CURE PASSIVATION',
  'SHIP LOT',         // ← violation: shipped before wafer sort test
  'WAFER SORT TEST',
  'FINAL LOT RELEASE',
]

const ANOMALY_STEP = 'SHIP LOT'
const VIOLATED_RULE = 'RULE_SHIP_BEFORE_TEST'

type HybridStatus = 'loading' | 'done' | 'error'

const IMPROVEMENTS = [
  { task: 'Task 1 Top-1', before: '87.7%', after: '93.7%', delta: '+6pp', color: '#00d4ff' },
  { task: 'Task 3 Anomaly', before: '60.8%', after: '100%', delta: '+39pp', color: '#00cc66' },
  { task: 'Task 2 Rule Validity', before: '—', after: '100%', delta: '✓', color: '#a78bfa' },
]

export default function LandingPage() {
  const navigate = useNavigate()
  const [hybridStatus, setHybridStatus] = useState<HybridStatus>('loading')
  const [hybridRule, setHybridRule] = useState<string>(VIOLATED_RULE)
  const [hybridIsValid, setHybridIsValid] = useState<boolean>(false)

  useEffect(() => {
    postInferAnomaly(DEMO_SEQUENCE, 'MOSFET')
      .then((res) => {
        setHybridIsValid(res.is_valid ?? false)
        setHybridRule(res.predicted_rule ?? VIOLATED_RULE)
        setHybridStatus('done')
      })
      .catch(() => {
        // fall back to pre-computed result
        setHybridIsValid(false)
        setHybridRule(VIOLATED_RULE)
        setHybridStatus('error')
      })
  }, [])

  return (
    <div className="max-w-5xl mx-auto space-y-8 py-2">

      {/* Header */}
      <div className="space-y-1">
        <div className="text-white/40 text-xs font-mono uppercase tracking-wider">AI Factory Austria · Infineon Track</div>
        <h1 className="text-3xl font-bold text-white leading-tight">
          Neuro-symbolic process sequence learning
        </h1>
        <p className="text-white/50 text-sm font-mono">
          LLM decoder (580K params) + deterministic physics rule engine (10 rules) · IC / IGBT / MOSFET fabrication
        </p>
      </div>

      {/* Improvement strip */}
      <div className="grid grid-cols-3 gap-3">
        {IMPROVEMENTS.map(({ task, before, after, delta, color }) => (
          <div
            key={task}
            className="rounded-xl border p-4"
            style={{ backgroundColor: '#0d1426', borderColor: `${color}28` }}
          >
            <div className="text-white/40 text-xs font-mono uppercase tracking-wider mb-2">{task}</div>
            <div className="flex items-end gap-2">
              <div className="font-mono text-lg text-white/30 line-through">{before}</div>
              <div className="font-mono text-2xl font-bold" style={{ color }}>{after}</div>
              <div className="font-mono text-xs pb-1" style={{ color: `${color}90` }}>{delta}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Main comparison */}
      <div className="space-y-3">
        <div className="text-white/40 text-xs font-mono uppercase tracking-wider">
          Same anomalous sequence — two systems
        </div>

        {/* Sequence preview */}
        <div className="rounded-xl border p-4 font-mono text-xs" style={{ backgroundColor: '#060b18', borderColor: '#ffffff10' }}>
          <div className="text-white/25 mb-2">MOSFET · 13-step sequence · RULE_SHIP_BEFORE_TEST violation</div>
          <div className="flex flex-wrap gap-1.5">
            {DEMO_SEQUENCE.map((step, i) => {
              const isAnomalous = step === ANOMALY_STEP
              return (
                <span
                  key={i}
                  className="px-2 py-0.5 rounded text-xs"
                  style={
                    isAnomalous
                      ? { backgroundColor: '#ff444425', border: '1px solid #ff444460', color: '#ff6666' }
                      : { backgroundColor: '#0d1426', border: '1px solid #ffffff12', color: '#ffffff50' }
                  }
                >
                  {isAnomalous ? `⚠ ${step}` : step}
                </span>
              )
            })}
          </div>
        </div>

        {/* Side-by-side verdict */}
        <div className="grid grid-cols-2 gap-4">

          {/* Left — encoder alone */}
          <div className="rounded-xl border p-5 space-y-4" style={{ backgroundColor: '#0d1426', borderColor: '#ffaa0028' }}>
            <div>
              <div className="font-mono text-sm font-bold text-white">Encoder alone</div>
              <div className="text-white/40 text-xs font-mono mt-0.5">ML only — no physics rules</div>
            </div>
            <div className="flex items-center gap-2">
              <div
                className="text-xs font-mono px-2 py-0.5 rounded"
                style={{ backgroundColor: '#ffaa0015', border: '1px solid #ffaa0030', color: '#ffaa00' }}
              >
                60.8% binary acc
              </div>
              <div className="text-white/30 text-xs font-mono">classifies everything as valid</div>
            </div>
            <div
              className="rounded-lg px-4 py-4 text-center"
              style={{ backgroundColor: '#00cc6608', border: '1px solid #00cc6620' }}
            >
              <div className="font-mono text-2xl font-bold" style={{ color: '#00cc6660' }}>✓ VALID</div>
              <div className="text-xs font-mono mt-2" style={{ color: '#ffffff25' }}>misclassified — model collapses to all-valid</div>
            </div>
          </div>

          {/* Right — physics hybrid */}
          <div className="rounded-xl border p-5 space-y-4" style={{ backgroundColor: '#0d1426', borderColor: '#00cc6628' }}>
            <div>
              <div className="font-mono text-sm font-bold text-white">Physics Hybrid</div>
              <div className="text-white/40 text-xs font-mono mt-0.5">decoder + 10 deterministic rules</div>
            </div>
            <div className="flex items-center gap-2">
              <div
                className="text-xs font-mono px-2 py-0.5 rounded"
                style={{ backgroundColor: '#00cc6615', border: '1px solid #00cc6630', color: '#00cc66' }}
              >
                100% binary acc
              </div>
              <div className="text-white/30 text-xs font-mono">
                {hybridStatus === 'loading' ? 'running live…' : hybridStatus === 'error' ? 'pre-computed' : 'live result'}
              </div>
            </div>

            {hybridStatus === 'loading' ? (
              <div className="rounded-lg px-4 py-4 text-center" style={{ backgroundColor: '#0a0f1e', border: '1px solid #ffffff08' }}>
                <div className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4 text-white/30" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  <span className="text-white/30 text-xs font-mono">calling live model…</span>
                </div>
              </div>
            ) : (
              <div
                className="rounded-lg px-4 py-4 text-center space-y-2"
                style={
                  hybridIsValid
                    ? { backgroundColor: '#00cc6608', border: '1px solid #00cc6640' }
                    : { backgroundColor: '#ff444412', border: '2px solid #ff444440' }
                }
              >
                <div
                  className="font-mono text-2xl font-bold"
                  style={{ color: hybridIsValid ? '#00cc66' : '#ff6666' }}
                >
                  {hybridIsValid ? '✓ VALID' : '⚠ ANOMALY DETECTED'}
                </div>
                {!hybridIsValid && (
                  <div
                    className="inline-block text-xs font-mono px-3 py-1 rounded"
                    style={{ backgroundColor: '#ff444420', border: '1px solid #ff444440', color: '#ff8888' }}
                  >
                    {hybridRule}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Architecture note */}
      <div
        className="rounded-xl border p-4 font-mono text-xs"
        style={{ backgroundColor: '#0d1426', borderColor: '#ffffff10' }}
      >
        <div className="flex flex-wrap items-center gap-3 text-white/50">
          <div className="text-white/30">Input</div>
          <div className="text-white/15">→</div>
          <div className="text-blue-400">Decoder <span className="text-white/30">(LlamaForCausalLM · 580K params)</span></div>
          <div className="text-white/15">→</div>
          <div className="text-purple-400">Physics engine <span className="text-white/30">(10 deterministic rules)</span></div>
          <div className="text-white/15">→</div>
          <div className="text-white/30">Task 1 re-ranked · Task 2 valid · Task 3 verdict</div>
        </div>
      </div>

      {/* Nav links */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'Live Inference', sub: 'upload CSV · run any sequence', path: '/infer', color: '#00cc66' },
          { label: 'Model Race', sub: 'all metrics · radar · summary table', path: '/models', color: '#00d4ff' },
          { label: 'Eval Browser', sub: '600 completions · 987 anomalies', path: '/batches', color: '#a78bfa' },
        ].map(({ label, sub, path, color }) => (
          <button
            key={path}
            onClick={() => navigate(path)}
            className="rounded-xl border p-4 text-left transition-all hover:opacity-80"
            style={{ backgroundColor: '#0d1426', borderColor: `${color}25` }}
          >
            <div className="font-mono font-bold text-white text-sm">{label}</div>
            <div className="text-white/35 text-xs font-mono mt-0.5">{sub}</div>
          </button>
        ))}
      </div>

    </div>
  )
}
