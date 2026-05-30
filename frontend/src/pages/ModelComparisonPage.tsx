import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis,
} from 'recharts'

const MODELS = [
  {
    id: 'gpt',
    name: 'GPT (autoregressive)',
    checkpoint: '001-gpt-fab',
    accent: '#00d4ff',
    params: '38M',
    size: '456 MB',
    task: 'Next-token prediction',
    how: 'Trained to predict each step given all previous steps. High perplexity on a step = that step is unlikely in this position.',
    stats: [
      { label: 'Top-1 accuracy', value: '81.4%' },
      { label: 'Top-5 accuracy', value: '99.99%' },
      { label: 'Perplexity', value: '1.373' },
      { label: 'Memorization probe', value: '36.1×' },
      { label: 'Parameters', value: '38M' },
      { label: 'Checkpoint size', value: '456 MB' },
      { label: 'Training epochs', value: '15' },
    ],
    note: 'Strongest memorization probe — 36.1× means the model assigns 36× higher loss to perturbed sequences vs real ones. Best signal for anomaly detection.',
  },
  {
    id: 'bert',
    name: 'BERT MLM (masked)',
    checkpoint: '002',
    accent: '#ff6b6b',
    params: '4.87M',
    size: '58 MB',
    task: 'Masked token prediction',
    how: 'Trained to fill in randomly masked steps. At inference, each step is masked one at a time — high fill-in loss means the model didn\'t expect that step here.',
    stats: [
      { label: 'Masked token acc.', value: '89.4%' },
      { label: 'Anomaly recall', value: '99.1%' },
      { label: 'Anomaly precision', value: '62.3%' },
      { label: 'F1 score', value: '76.5%' },
      { label: 'Overall accuracy', value: '62.1%' },
      { label: 'Parameters', value: '4.87M' },
      { label: 'Checkpoint size', value: '58 MB' },
    ],
    note: 'Used for the actual anomaly detection task. Calibrated thresholds: p95 step loss = 0.793, OOD mean loss = 0.252. Catches 978/987 defective sequences.',
  },
  {
    id: 'lstm',
    name: 'LSTM (baseline)',
    checkpoint: '005-lstm-baseline',
    accent: '#00cc66',
    params: '3.55M',
    size: '42 MB',
    task: 'Next-token prediction',
    how: '2-layer LSTM with hidden size 512. Same training objective as GPT. Included as a baseline to measure how much the transformer architecture actually helps.',
    stats: [
      { label: 'Top-1 accuracy', value: '81.3%' },
      { label: 'Top-5 accuracy', value: '99.99%' },
      { label: 'Perplexity', value: '1.375' },
      { label: 'Memorization probe', value: '24.8×' },
      { label: 'Parameters', value: '3.55M' },
      { label: 'Checkpoint size', value: '42 MB' },
      { label: 'Training epochs', value: '9' },
    ],
    note: 'Almost identical accuracy to GPT (81.3% vs 81.4%) at 1/10th the size. Lower memorization probe (24.8×) means weaker anomaly signal. Good baseline.',
  },
]

const RADAR_DATA = [
  { axis: 'Accuracy', GPT: 81, LSTM: 81, BERT: 62 },
  { axis: 'Anomaly recall', GPT: 36, LSTM: 25, BERT: 99 },
  { axis: 'Compactness', GPT: 8, LSTM: 92, BERT: 88 },
  { axis: 'Probe ratio', GPT: 100, LSTM: 69, BERT: 50 },
  { axis: 'Precision', GPT: 60, LSTM: 50, BERT: 62 },
]

export default function ModelComparisonPage() {
  return (
    <div className="max-w-6xl mx-auto space-y-8 py-2">

      <div>
        <h1 className="text-white font-bold text-2xl">Model Comparison</h1>
        <p className="text-white/40 text-xs font-mono mt-1">
          Three architectures, same dataset: 150k Infineon sequences across IC, IGBT, MOSFET routes. 208-token vocabulary.
        </p>
      </div>

      {/* Model cards */}
      <div className="grid grid-cols-3 gap-5">
        {MODELS.map((m) => (
          <div
            key={m.id}
            className="rounded-xl border p-4 space-y-3"
            style={{ backgroundColor: '#0d1426', borderColor: `${m.accent}28` }}
          >
            <div>
              <div className="font-mono font-bold text-white text-base">{m.name}</div>
              <div className="text-white/30 text-xs font-mono mt-0.5">checkpoint: {m.checkpoint}</div>
            </div>

            <div className="rounded-lg p-3 text-xs font-mono leading-relaxed text-white/50" style={{ backgroundColor: '#060b18' }}>
              <span className="text-white/70">Task: </span>{m.task}<br />
              <span className="text-white/70">How: </span>{m.how}
            </div>

            <div className="space-y-1">
              {m.stats.map((s) => (
                <div key={s.label} className="flex justify-between text-xs font-mono py-1 border-b border-white/5 last:border-0">
                  <span className="text-white/40">{s.label}</span>
                  <span className="text-white font-bold">{s.value}</span>
                </div>
              ))}
            </div>

            <div className="text-xs font-mono leading-relaxed text-white/40 pt-1 border-t border-white/5">
              {m.note}
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-5">
        <div className="rounded-xl border p-4 space-y-3" style={{ backgroundColor: '#0d1426', borderColor: '#ffffff10' }}>
          <div className="font-mono text-sm text-white">Accuracy vs anomaly recall</div>
          <div className="text-white/30 text-xs font-mono">Accuracy = correct next-step predictions on held-out test set.</div>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart
              data={[
                { name: 'GPT', 'Step accuracy': 81.4, 'Anomaly recall': 36 },
                { name: 'BERT', 'Step accuracy': 62.1, 'Anomaly recall': 99.1 },
                { name: 'LSTM', 'Step accuracy': 81.3, 'Anomaly recall': 24.8 },
              ]}
              margin={{ top: 4, right: 10, left: -10, bottom: 4 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
              <XAxis dataKey="name" tick={{ fill: '#ffffff60', fontSize: 11, fontFamily: 'monospace' }} />
              <YAxis tick={{ fill: '#ffffff60', fontSize: 10, fontFamily: 'monospace' }} domain={[0, 110]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#111827', border: '1px solid #ffffff20', fontFamily: 'monospace', fontSize: 11 }}
                formatter={(v: number, name: string) => [`${v.toFixed(1)}%`, name]}
              />
              <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#ffffff60' }} />
              <Bar dataKey="Step accuracy" fill="#00d4ff" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Anomaly recall" fill="#ff6b6b" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="rounded-xl border p-4 space-y-3" style={{ backgroundColor: '#0d1426', borderColor: '#ffffff10' }}>
          <div className="font-mono text-sm text-white">Capability radar</div>
          <div className="text-white/30 text-xs font-mono">Normalized. Probe ratio normalized to GPT=100.</div>
          <ResponsiveContainer width="100%" height={210}>
            <RadarChart data={RADAR_DATA} margin={{ top: 10, right: 30, left: 30, bottom: 10 }}>
              <PolarGrid stroke="#ffffff12" />
              <PolarAngleAxis dataKey="axis" tick={{ fill: '#ffffff50', fontSize: 9, fontFamily: 'monospace' }} />
              <Radar name="GPT" dataKey="GPT" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.12} />
              <Radar name="BERT" dataKey="BERT" stroke="#ff6b6b" fill="#ff6b6b" fillOpacity={0.12} />
              <Radar name="LSTM" dataKey="LSTM" stroke="#00cc66" fill="#00cc66" fillOpacity={0.12} />
              <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#ffffff60' }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Size vs accuracy table */}
      <div className="rounded-xl border overflow-hidden" style={{ backgroundColor: '#0d1426', borderColor: '#ffffff10' }}>
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="border-b border-white/10 bg-[#060b18]">
              {['Model', 'Params', 'Size', 'Top-1 acc.', 'Anomaly recall', 'Probe ratio', 'Best for'].map((h) => (
                <th key={h} className="text-left px-4 py-3 text-white/40 font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-white/5">
              <td className="px-4 py-3 text-white font-bold">GPT</td>
              <td className="px-4 py-3 text-white/70">38M</td>
              <td className="px-4 py-3 text-white/70">456 MB</td>
              <td className="px-4 py-3 text-white/70">81.4%</td>
              <td className="px-4 py-3 text-white/40">—</td>
              <td className="px-4 py-3 text-white/70">36.1×</td>
              <td className="px-4 py-3 text-white/50">Process sequence modeling</td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="px-4 py-3 text-white font-bold">BERT</td>
              <td className="px-4 py-3 text-white/70">4.87M</td>
              <td className="px-4 py-3 text-white/70">58 MB</td>
              <td className="px-4 py-3 text-white/40">— (MLM task)</td>
              <td className="px-4 py-3" style={{ color: '#ff6b6b' }}>99.1%</td>
              <td className="px-4 py-3 text-white/40">—</td>
              <td className="px-4 py-3 text-white/50">Anomaly detection (deployed)</td>
            </tr>
            <tr>
              <td className="px-4 py-3 text-white font-bold">LSTM</td>
              <td className="px-4 py-3 text-white/70">3.55M</td>
              <td className="px-4 py-3 text-white/70">42 MB</td>
              <td className="px-4 py-3 text-white/70">81.3%</td>
              <td className="px-4 py-3 text-white/40">—</td>
              <td className="px-4 py-3 text-white/70">24.8×</td>
              <td className="px-4 py-3 text-white/50">Baseline / edge deployment</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="text-white/30 text-xs font-mono">
        All models trained on Leonardo HPC (CINECA), 4× A100 64GB GPUs. GPT and LSTM share the same autoregressive objective; BERT uses masked language modeling.
        Anomaly recall/precision figures are for BERT only, evaluated on 1,587 Infineon eval sequences (987 invalid, 600 partial-valid).
      </div>
    </div>
  )
}
