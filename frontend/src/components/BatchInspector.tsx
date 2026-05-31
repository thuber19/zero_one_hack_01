import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getCompletions, getAnomalies } from '../lib/api'
import { useAppStore } from '../lib/store'
import type { CompletionListResponse, AnomalyListResponse } from '../types/api'

const PAGE_SIZE = 20

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-white/60 tabular-nums">{(value * 100).toFixed(0)}%</span>
      <div className="w-10 h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div className="h-full rounded-full bg-[#00d4ff]" style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  )
}

const FAMILY_COLORS: Record<string, string> = {
  IC: '#00d4ff',
  IGBT: '#ffaa00',
  MOSFET: '#00cc66',
}

function FamilyBadge({ family }: { family: string }) {
  const color = FAMILY_COLORS[family] ?? '#ffffff60'
  return (
    <span
      className="inline-block px-2 py-0.5 rounded text-xs font-mono font-semibold"
      style={{ color, backgroundColor: `${color}20`, border: `1px solid ${color}40` }}
    >
      {family}
    </span>
  )
}

type SortOrder = 'asc' | 'desc'

type CompletionSortField = 'example_id' | 'family' | 'completion_fraction' | 'rank1' | 'partial_step_count'
type AnomalySortField = 'example_id' | 'family' | 'is_valid' | 'score' | 'predicted_rule' | 'sequence_step_count'

function SortIndicator({ active, order }: { active: boolean; order: SortOrder }) {
  if (!active) return null
  return <span className="ml-1 text-[#00d4ff]">{order === 'asc' ? '↑' : '↓'}</span>
}

export function CompletionsTable() {
  const navigate = useNavigate()
  const setActiveSequence = useAppStore((s) => s.setActiveSequence)

  const [page, setPage] = useState(1)
  const [sortBy, setSortBy] = useState<CompletionSortField>('completion_fraction')
  const [order, setOrder] = useState<SortOrder>('desc')

  const { data, isLoading, isError } = useQuery<CompletionListResponse>({
    queryKey: ['completions', page, PAGE_SIZE, sortBy, order],
    queryFn: () => getCompletions({ page, page_size: PAGE_SIZE, sort_by: sortBy, order }),
    placeholderData: (prev) => prev,
    retry: 1,
  })

  function handleSort(field: CompletionSortField) {
    if (sortBy === field) {
      setOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortBy(field)
      setOrder('desc')
    }
    setPage(1)
  }

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 1

  const columns: { key: CompletionSortField; label: string }[] = [
    { key: 'example_id', label: 'Example ID' },
    { key: 'family', label: 'Family' },
    { key: 'completion_fraction', label: 'Completion' },
    { key: 'rank1', label: 'Rank 1 Prediction' },
    { key: 'partial_step_count', label: 'Steps' },
  ]

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto rounded-lg border border-white/10">
        <table className="w-full text-sm font-mono">
          <thead>
            <tr className="border-b border-white/10 bg-[#0d1426]">
              {columns.map(({ key, label }) => (
                <th
                  key={key}
                  onClick={() => handleSort(key)}
                  className="text-left px-3 py-3 text-white/50 text-xs uppercase tracking-wide cursor-pointer hover:text-white transition-colors select-none"
                >
                  {label}
                  <SortIndicator active={sortBy === key} order={order} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading && Array.from({ length: 5 }).map((_, i) => (
              <tr key={i} className="border-b border-white/5">
                {[55, 20, 35, 70, 15].map((w, j) => (
                  <td key={j} className="px-3 py-3">
                    <div className="h-3 rounded bg-white/5 animate-pulse" style={{ width: `${w}%` }} />
                  </td>
                ))}
              </tr>
            ))}
            {isError && (
              <tr>
                <td colSpan={5} className="text-center py-8 text-[#ff4444]/70">
                  Failed to load — API not running?
                </td>
              </tr>
            )}
            {!isLoading && !isError && data?.items.map((item) => (
              <tr
                key={item.example_id}
                onClick={() => {
                  setActiveSequence(item.example_id, 'completion')
                  navigate('/journey')
                }}
                className="border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors"
              >
                <td className="px-3 py-2.5 text-[#00d4ff]">{item.example_id}</td>
                <td className="px-3 py-2.5"><FamilyBadge family={item.family} /></td>
                <td className="px-3 py-2.5"><ProgressBar value={item.completion_fraction} /></td>
                <td className="px-3 py-2.5 text-white/70 truncate max-w-[200px]">
                  {item.rank1.length > 28 ? item.rank1.slice(0, 28) + '…' : item.rank1}
                </td>
                <td className="px-3 py-2.5 text-white/60">{item.partial_step_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs font-mono text-white/40">
          <span>Page {page} of {totalPages} ({data?.total ?? 0} total completions)</span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1 rounded border border-white/10 hover:border-[#00d4ff40] disabled:opacity-30 transition-colors"
            >
              Prev
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="px-3 py-1 rounded border border-white/10 hover:border-[#00d4ff40] disabled:opacity-30 transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export function AnomaliesTable() {
  const navigate = useNavigate()
  const setActiveSequence = useAppStore((s) => s.setActiveSequence)

  const [page, setPage] = useState(1)
  const [sortBy, setSortBy] = useState<AnomalySortField>('is_valid')
  const [order, setOrder] = useState<SortOrder>('asc')

  const { data, isLoading, isError } = useQuery<AnomalyListResponse>({
    queryKey: ['anomalies', page, PAGE_SIZE, sortBy, order],
    queryFn: () => getAnomalies({ page, page_size: PAGE_SIZE, sort_by: sortBy, order }),
    placeholderData: (prev) => prev,
    retry: 1,
  })

  function handleSort(field: AnomalySortField) {
    if (sortBy === field) {
      setOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortBy(field)
      setOrder('asc')
    }
    setPage(1)
  }

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 1

  const columns: { key: AnomalySortField; label: string }[] = [
    { key: 'example_id', label: 'Example ID' },
    { key: 'family', label: 'Family' },
    { key: 'is_valid', label: 'Status' },
    { key: 'score', label: 'Score' },
    { key: 'predicted_rule', label: 'Rule' },
    { key: 'sequence_step_count', label: 'Steps' },
  ]

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto rounded-lg border border-white/10">
        <table className="w-full text-sm font-mono">
          <thead>
            <tr className="border-b border-white/10 bg-[#0d1426]">
              {columns.map(({ key, label }) => (
                <th
                  key={key}
                  onClick={() => handleSort(key)}
                  className="text-left px-3 py-3 text-white/50 text-xs uppercase tracking-wide cursor-pointer hover:text-white transition-colors select-none"
                >
                  {label}
                  <SortIndicator active={sortBy === key} order={order} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading && Array.from({ length: 5 }).map((_, i) => (
              <tr key={i} className="border-b border-white/5">
                {[55, 20, 25, 20, 50, 15].map((w, j) => (
                  <td key={j} className="px-3 py-3">
                    <div className="h-3 rounded bg-white/5 animate-pulse" style={{ width: `${w}%` }} />
                  </td>
                ))}
              </tr>
            ))}
            {isError && (
              <tr>
                <td colSpan={6} className="text-center py-8 text-[#ff4444]/70">
                  Failed to load — API not running?
                </td>
              </tr>
            )}
            {!isLoading && !isError && data?.items.map((item) => (
              <tr
                key={item.example_id}
                onClick={() => {
                  setActiveSequence(item.example_id, 'anomaly')
                  navigate('/journey')
                }}
                className="border-b border-white/5 hover:bg-white/5 cursor-pointer transition-colors"
              >
                <td className="px-3 py-2.5 text-[#00d4ff]">{item.example_id}</td>
                <td className="px-3 py-2.5"><FamilyBadge family={item.family} /></td>
                <td className="px-3 py-2.5">
                  {item.is_valid ? (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold text-[#00cc66] bg-[#00cc6620] border border-[#00cc6640]">
                      ✓ Valid
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold text-[#ff4444] bg-[#ff444420] border border-[#ff444440]">
                      ⚠ Anomaly
                    </span>
                  )}
                </td>
                <td className={`px-3 py-2.5 ${item.is_valid === false ? 'text-[#ff4444]' : 'text-white/60'}`}>
                  {item.score.toFixed(3)}
                </td>
                <td className="px-3 py-2.5">
                  {item.predicted_rule ? (
                    <span className="inline-block px-2 py-0.5 rounded text-xs font-mono text-[#ff4444] bg-[#ff444415] border border-[#ff444430]">
                      {item.predicted_rule}
                    </span>
                  ) : (
                    <span className="text-white/30">—</span>
                  )}
                </td>
                <td className="px-3 py-2.5 text-white/60">{item.sequence_step_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs font-mono text-white/40">
          <span>Page {page} of {totalPages} ({data?.total ?? 0} total anomalies)</span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1 rounded border border-white/10 hover:border-[#ff444440] disabled:opacity-30 transition-colors"
            >
              Prev
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="px-3 py-1 rounded border border-white/10 hover:border-[#ff444440] disabled:opacity-30 transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
