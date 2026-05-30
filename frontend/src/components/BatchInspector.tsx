import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getBatches, getBatchDetail } from '../lib/api'
import { FALLBACK_BATCHES } from '../hooks/useBatches'
import type { BatchSummary, BatchListResponse, PredictResponse } from '../types/api'

type SortField = 'batch_id' | 'timestamp' | 'material' | 'predicted_yield' | 'defect_probability' | 'confidence' | 'risk_steps_detected'
type SortOrder = 'asc' | 'desc'

function grade(yield_: number): { letter: string; color: string; bg: string } {
  if (yield_ >= 0.90) return { letter: 'A', color: '#00cc66', bg: '#00cc6620' }
  if (yield_ >= 0.75) return { letter: 'B', color: '#00d4ff', bg: '#00d4ff20' }
  if (yield_ >= 0.60) return { letter: 'C', color: '#ffaa00', bg: '#ffaa0020' }
  return { letter: 'F', color: '#ff4444', bg: '#ff444420' }
}

function RiskBar({ value }: { value: number }) {
  const color = value >= 0.8 ? '#ff4444' : value >= 0.5 ? '#ffaa00' : '#334155'
  return (
    <div className="flex items-center gap-2">
      <span className={value >= 0.8 ? 'text-risk-red' : 'text-white/60'}>
        {(value * 100).toFixed(1)}%
      </span>
      <div className="w-12 h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

const FAMILY_ICONS: Record<string, string> = {
  IC: '🔷',
  IGBT: '⚡',
  MOSFET: '🔋',
}

interface DrawerProps {
  batchId: string
  onClose: () => void
}

function BatchDetailDrawer({ batchId, onClose }: DrawerProps) {
  const { data, isLoading } = useQuery<PredictResponse>({
    queryKey: ['batch-detail', batchId],
    queryFn: () => getBatchDetail(batchId),
    staleTime: Infinity,
    retry: 1,
  })

  const g = data ? grade(data.predicted_yield) : null

  return (
    <div className="fixed inset-y-0 right-0 w-[420px] bg-[#0a0f1e] border-l border-white/10 z-50 overflow-y-auto shadow-2xl">
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <span className="font-mono text-sm text-accent">{batchId}</span>
        <button onClick={onClose} className="text-white/40 hover:text-white text-xl leading-none">&times;</button>
      </div>
      <div className="p-4">
        {isLoading && <div className="text-white/30 font-mono text-sm">Loading...</div>}
        {data && g && (
          <div className="space-y-4">
            {/* Grade badge */}
            <div className="flex items-center gap-3">
              <div
                className="w-14 h-14 rounded-xl flex items-center justify-center text-3xl font-bold font-mono"
                style={{ color: g.color, backgroundColor: g.bg, border: `2px solid ${g.color}44` }}
              >
                {g.letter}
              </div>
              <div>
                <div className="text-white font-bold text-lg font-mono">
                  Quality Score: {(data.predicted_yield * 100).toFixed(1)}%
                </div>
                <div className="text-white/40 text-xs font-mono mt-0.5">
                  {data.risk_steps_detected} issues found · {(data.confidence * 100).toFixed(0)}% confidence
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Quality Score', value: `${(data.predicted_yield * 100).toFixed(1)}%` },
                { label: 'Confidence', value: `${(data.confidence * 100).toFixed(0)}%` },
                { label: 'Issues Found', value: String(data.risk_steps_detected) },
                { label: 'Flagged', value: data.anomalous_batches > 0 ? '🚨 Yes' : '✅ No' },
              ].map(({ label, value }) => (
                <div key={label} className="bg-[#0d1426] rounded p-3 border border-white/10">
                  <div className="text-white/40 text-xs font-mono">{label}</div>
                  <div className="text-white font-mono text-base mt-1">{value}</div>
                </div>
              ))}
            </div>

            {data.per_step && data.per_step.length > 0 && (
              <div>
                <div className="text-white/40 text-xs font-mono mb-2 uppercase tracking-wider">Step Risk Heatmap</div>
                <div className="flex flex-wrap gap-1">
                  {data.per_step.map((step) => (
                    <div
                      key={step.step_id}
                      title={`${step.step_name}: ${(step.risk_score * 100).toFixed(0)}% failure risk`}
                      className="w-3 h-3 rounded-sm transition-colors"
                      style={{
                        backgroundColor:
                          step.risk_score >= 0.85 ? '#ff4444' :
                          step.risk_score >= 0.70 ? '#ffaa00' : '#1e2a3a',
                      }}
                    />
                  ))}
                </div>
                <div className="flex gap-4 mt-2 text-xs font-mono text-white/30">
                  <span><span className="inline-block w-2 h-2 rounded-sm bg-[#ff4444] mr-1" />High risk</span>
                  <span><span className="inline-block w-2 h-2 rounded-sm bg-[#ffaa00] mr-1" />Medium</span>
                  <span><span className="inline-block w-2 h-2 rounded-sm bg-[#1e2a3a] border border-white/20 mr-1" />Normal</span>
                </div>
              </div>
            )}
          </div>
        )}
        {!isLoading && !data && (
          <div className="text-white/30 font-mono text-sm">No detail data available</div>
        )}
      </div>
    </div>
  )
}

const COLUMNS: { key: SortField; label: string }[] = [
  { key: 'batch_id', label: 'Batch ID' },
  { key: 'material', label: 'Type' },
  { key: 'predicted_yield', label: 'Quality Score' },
  { key: 'defect_probability', label: 'Failure Risk' },
  { key: 'confidence', label: 'Confidence' },
  { key: 'risk_steps_detected', label: 'Issues Found' },
]

export default function BatchInspector() {
  const [page, setPage] = useState(1)
  const [sortBy, setSortBy] = useState<SortField>('defect_probability')
  const [order, setOrder] = useState<SortOrder>('desc')
  const [selectedBatchId, setSelectedBatchId] = useState<string | null>(null)
  const PAGE_SIZE = 20

  const { data, isError } = useQuery<BatchListResponse>({
    queryKey: ['batches', page, PAGE_SIZE, sortBy, order],
    queryFn: () => getBatches({ page, page_size: PAGE_SIZE, sort_by: sortBy, order }),
    placeholderData: (prev) => prev,
    retry: 1,
  })

  const listData = isError || !data ? FALLBACK_BATCHES : data

  function handleSort(field: SortField) {
    if (sortBy === field) {
      setOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortBy(field)
      setOrder('desc')
    }
    setPage(1)
  }

  const totalPages = Math.ceil(listData.total / PAGE_SIZE)

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto rounded-lg border border-white/10">
        <table className="w-full text-sm font-mono">
          <thead>
            <tr className="border-b border-white/10 bg-[#0d1426]">
              <th className="text-left px-3 py-3 text-white/50 text-xs uppercase tracking-wide w-8">Grade</th>
              {COLUMNS.map(({ key, label }) => (
                <th
                  key={key}
                  onClick={() => handleSort(key)}
                  className="text-left px-3 py-3 text-white/50 text-xs uppercase tracking-wide cursor-pointer hover:text-white transition-colors select-none"
                >
                  {label}
                  {sortBy === key && (
                    <span className="ml-1 text-accent">{order === 'asc' ? '↑' : '↓'}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {listData.batches.length === 0 && (
              <tr>
                <td colSpan={7} className="text-center py-8 text-white/30">No batches found</td>
              </tr>
            )}
            {listData.batches.map((batch: BatchSummary) => {
              const g = grade(batch.predicted_yield)
              return (
                <tr
                  key={batch.batch_id}
                  onClick={() => setSelectedBatchId(batch.batch_id)}
                  className={`border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors ${
                    batch.defect_probability >= 0.8 ? 'border-l-2 border-l-risk-red' : ''
                  }`}
                >
                  <td className="px-3 py-2.5">
                    <span
                      className="inline-block w-6 h-6 rounded text-center text-xs font-bold leading-6"
                      style={{ color: g.color, backgroundColor: g.bg }}
                    >
                      {g.letter}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-accent">{batch.batch_id}</td>
                  <td className="px-3 py-2.5 text-white/70">
                    {FAMILY_ICONS[batch.material] ?? '🔷'} {batch.material}
                  </td>
                  <td className={`px-3 py-2.5 ${
                    batch.predicted_yield >= 0.85 ? 'text-yield-green' :
                    batch.predicted_yield >= 0.70 ? 'text-risk-amber' : 'text-risk-red'
                  }`}>
                    {(batch.predicted_yield * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-2.5">
                    <RiskBar value={batch.defect_probability} />
                  </td>
                  <td className="px-3 py-2.5 text-white/60">{(batch.confidence * 100).toFixed(0)}%</td>
                  <td className={`px-3 py-2.5 ${batch.risk_steps_detected > 10 ? 'text-risk-red' : batch.risk_steps_detected > 3 ? 'text-risk-amber' : 'text-white/60'}`}>
                    {batch.risk_steps_detected > 0 ? `⚠ ${batch.risk_steps_detected}` : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs font-mono text-white/40">
          <span>Page {page} of {totalPages} ({listData.total} total wafers)</span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1 rounded border border-white/10 hover:border-accent/40 disabled:opacity-30 transition-colors"
            >
              Prev
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="px-3 py-1 rounded border border-white/10 hover:border-accent/40 disabled:opacity-30 transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {selectedBatchId && (
        <>
          <div className="fixed inset-0 bg-black/40 z-40" onClick={() => setSelectedBatchId(null)} />
          <BatchDetailDrawer batchId={selectedBatchId} onClose={() => setSelectedBatchId(null)} />
        </>
      )}
    </div>
  )
}
