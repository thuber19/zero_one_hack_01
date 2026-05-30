import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getBatches, getBatchDetail } from '../lib/api'
import { FALLBACK_BATCHES } from '../hooks/useBatches'
import type { BatchSummary, BatchListResponse, PredictResponse } from '../types/api'

type SortField = 'batch_id' | 'timestamp' | 'material' | 'predicted_yield' | 'defect_probability' | 'confidence' | 'risk_steps_detected'
type SortOrder = 'asc' | 'desc'

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

  return (
    <div className="fixed inset-y-0 right-0 w-[400px] bg-[#0a0f1e] border-l border-white/10 z-50 overflow-y-auto shadow-2xl">
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <span className="font-mono text-sm text-accent">{batchId}</span>
        <button onClick={onClose} className="text-white/40 hover:text-white text-xl leading-none">&times;</button>
      </div>
      <div className="p-4">
        {isLoading && <div className="text-white/30 font-mono text-sm">Loading...</div>}
        {data && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Predicted Yield', value: `${(data.predicted_yield * 100).toFixed(1)}%` },
                { label: 'Confidence', value: `${(data.confidence * 100).toFixed(0)}%` },
                { label: 'Risk Steps', value: String(data.risk_steps_detected) },
                { label: 'Anomalous', value: String(data.anomalous_batches) },
              ].map(({ label, value }) => (
                <div key={label} className="bg-[#0d1426] rounded p-3 border border-white/10">
                  <div className="text-white/40 text-xs font-mono">{label}</div>
                  <div className="text-white font-mono text-lg mt-1">{value}</div>
                </div>
              ))}
            </div>
            {data.per_step && data.per_step.length > 0 && (
              <div>
                <div className="text-white/40 text-xs font-mono mb-2">Per-Step Risk</div>
                <div className="flex flex-wrap gap-1">
                  {data.per_step.map((step) => (
                    <div
                      key={step.step_id}
                      title={`${step.step_name}: ${step.risk_score.toFixed(3)}`}
                      className="w-3 h-3 rounded-sm"
                      style={{
                        backgroundColor:
                          step.risk_score >= 0.85 ? '#ff4444' :
                          step.risk_score >= 0.70 ? '#ffaa00' : '#1e2a3a',
                      }}
                    />
                  ))}
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
  { key: 'timestamp', label: 'Timestamp' },
  { key: 'material', label: 'Material' },
  { key: 'predicted_yield', label: 'Yield' },
  { key: 'defect_probability', label: 'Defect Prob.' },
  { key: 'confidence', label: 'Confidence' },
  { key: 'risk_steps_detected', label: 'Risk Steps' },
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
            {listData.batches.map((batch: BatchSummary) => (
              <tr
                key={batch.batch_id}
                onClick={() => setSelectedBatchId(batch.batch_id)}
                className={`border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors ${
                  batch.defect_probability >= 0.8 ? 'border-l-2 border-l-risk-red' : ''
                }`}
              >
                <td className="px-3 py-2.5 text-accent">{batch.batch_id}</td>
                <td className="px-3 py-2.5 text-white/60 text-xs">{new Date(batch.timestamp).toLocaleString()}</td>
                <td className="px-3 py-2.5 text-white/80">{batch.material}</td>
                <td className={`px-3 py-2.5 ${
                  batch.predicted_yield >= 0.85 ? 'text-yield-green' :
                  batch.predicted_yield >= 0.70 ? 'text-risk-amber' : 'text-risk-red'
                }`}>
                  {(batch.predicted_yield * 100).toFixed(1)}%
                </td>
                <td className={`px-3 py-2.5 ${batch.defect_probability >= 0.8 ? 'text-risk-red' : 'text-white/60'}`}>
                  {(batch.defect_probability * 100).toFixed(1)}%
                </td>
                <td className="px-3 py-2.5 text-white/60">{(batch.confidence * 100).toFixed(0)}%</td>
                <td className={`px-3 py-2.5 ${batch.risk_steps_detected > 10 ? 'text-risk-red' : batch.risk_steps_detected > 3 ? 'text-risk-amber' : 'text-white/60'}`}>
                  {batch.risk_steps_detected}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs font-mono text-white/40">
          <span>Page {page} of {totalPages} ({listData.total} total)</span>
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
