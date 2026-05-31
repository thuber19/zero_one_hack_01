import { useState, useRef } from 'react'
import { postInferNextStep, postInferComplete, postInferAnomaly } from '../lib/api'
import type { LiveInferResponse } from '../types/api'

type Task = 'nextstep' | 'complete' | 'anomaly'

interface CSVRow {
  exampleId: string
  family: string
  steps: string[]
  colType: 'PARTIAL_SEQUENCE' | 'SEQUENCE'
}

interface BatchResult {
  row: CSVRow
  status: 'pending' | 'running' | 'done' | 'error'
  response: LiveInferResponse | null
}

interface ParseResult {
  rows: CSVRow[]
  error?: string
}

function parseParts(line: string): string[] {
  return line.match(/(".*?"|[^,]+)/g)?.map((p) => p.trim().replace(/^"|"$/g, '')) ?? line.split(',').map((p) => p.trim())
}

function parseCSV(text: string): ParseResult {
  const lines = text.split('\n').filter((l) => l.trim())
  if (lines.length < 2) return { rows: [], error: 'File is empty or has only a header row.' }

  // Strip BOM from first line
  const headerLine = lines[0].replace(/^﻿/, '')
  const headers = parseParts(headerLine)

  const idCol      = headers.findIndex((h) => h === 'EXAMPLE_ID')
  const familyCol  = headers.findIndex((h) => h === 'FAMILY')
  const seqCol     = headers.findIndex((h) => h === 'PARTIAL_SEQUENCE')
  const fullSeqCol = headers.findIndex((h) => h === 'SEQUENCE')
  const predSeqCol = headers.findIndex((h) => h === 'PREDICTED_SEQUENCE')
  const seqIdCol   = headers.findIndex((h) => h === 'SEQUENCE_ID')
  const stepCol    = headers.findIndex((h) => h === 'STEP')

  // Format 1: PARTIAL_SEQUENCE / SEQUENCE / PREDICTED_SEQUENCE — pipe-delimited per row
  const pipeCol = seqCol !== -1 ? seqCol : fullSeqCol !== -1 ? fullSeqCol : predSeqCol !== -1 ? predSeqCol : -1
  if (pipeCol !== -1) {
    const colType: CSVRow['colType'] = seqCol !== -1 ? 'PARTIAL_SEQUENCE' : 'SEQUENCE'
    const rows: CSVRow[] = []
    for (let i = 1; i < lines.length; i++) {
      const parts = parseParts(lines[i])
      const seq = parts[pipeCol]?.trim() ?? ''
      if (!seq) continue
      rows.push({
        exampleId: idCol !== -1 ? (parts[idCol]?.trim() ?? `row_${i}`) : `row_${i}`,
        family: familyCol !== -1 ? (parts[familyCol]?.trim() ?? 'MOSFET') : 'MOSFET',
        steps: seq.split('|').map((s) => s.trim()).filter(Boolean),
        colType,
      })
    }
    return rows.length > 0 ? { rows } : { rows: [], error: 'No data rows found.' }
  }

  // Format 2: SEQUENCE_ID + STEP — one step per row, grouped by SEQUENCE_ID
  if (seqIdCol !== -1 && stepCol !== -1) {
    const grouped = new Map<string, string[]>()
    const order: string[] = []
    for (let i = 1; i < lines.length; i++) {
      const parts = parseParts(lines[i])
      const sid  = parts[seqIdCol]?.trim()
      const step = parts[stepCol]?.trim()
      if (!sid || !step) continue
      if (!grouped.has(sid)) { grouped.set(sid, []); order.push(sid) }
      grouped.get(sid)!.push(step)
    }
    const rows: CSVRow[] = order.map((sid) => ({
      exampleId: sid,
      family: 'MOSFET',
      steps: grouped.get(sid)!,
      colType: 'SEQUENCE',
    }))
    return rows.length > 0 ? { rows } : { rows: [], error: 'No data rows found.' }
  }

  // Format 3: STEP only — entire file is a single sequence (one step per row)
  if (stepCol !== -1) {
    const steps: string[] = []
    for (let i = 1; i < lines.length; i++) {
      const parts = parseParts(lines[i])
      const step = parts[stepCol]?.trim()
      if (step) steps.push(step)
    }
    if (steps.length > 0) {
      return { rows: [{ exampleId: 'sequence_1', family: 'MOSFET', steps, colType: 'SEQUENCE' }] }
    }
  }

  // Unrecognized format
  const found = headers.slice(0, 6).join(', ') + (headers.length > 6 ? ', …' : '')
  return {
    rows: [],
    error: `Unrecognized format. Need a PARTIAL_SEQUENCE, SEQUENCE, PREDICTED_SEQUENCE, SEQUENCE_ID+STEP, or STEP column. Found: ${found}`,
  }
}

function exportCSV(results: BatchResult[], task: Task) {
  const headers =
    task === 'nextstep' ? ['EXAMPLE_ID', 'FAMILY', 'RANK_1', 'RANK_2', 'RANK_3', 'RANK_4', 'RANK_5']
    : task === 'complete' ? ['EXAMPLE_ID', 'FAMILY', 'IS_VALID', 'PREDICTED_STEPS']
    : ['EXAMPLE_ID', 'FAMILY', 'IS_VALID', 'SCORE', 'PREDICTED_RULE']
  const rows = results.map(({ row, response }) => {
    if (!response) return [row.exampleId, row.family, ...headers.slice(2).map(() => '')]
    if (task === 'nextstep') {
      const p = response.predictions ?? []
      return [row.exampleId, row.family, p[0] ?? '', p[1] ?? '', p[2] ?? '', p[3] ?? '', p[4] ?? '']
    }
    if (task === 'complete') {
      return [row.exampleId, row.family, String(response.is_valid ?? ''), String((response.completion ?? []).length)]
    }
    return [row.exampleId, row.family, String(response.is_valid ?? ''), String(response.score ?? ''), response.predicted_rule ?? '']
  })
  const csv = [headers, ...rows].map((r) => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `results_${task}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

function ResultCell({ result, task }: { result: BatchResult; task: Task }) {
  if (result.status === 'pending') return <span className="text-white/15 font-mono text-xs">—</span>
  if (result.status === 'running') return (
    <span className="flex items-center gap-1.5 text-white/30 font-mono text-xs">
      <svg className="animate-spin h-3 w-3 flex-shrink-0" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      running
    </span>
  )
  if (result.status === 'error') return <span className="text-red-400/60 font-mono text-xs">error</span>
  const r = result.response!
  if (task === 'nextstep') {
    const p = r.predictions ?? []
    return (
      <span className="font-mono text-xs text-white/60 truncate block max-w-xs">
        <span className="text-white/80">{p[0]}</span>
        {p[1] && <span className="text-white/30"> · {p[1]}</span>}
        {p[2] && <span className="text-white/20"> · {p[2]}</span>}
      </span>
    )
  }
  if (task === 'complete') {
    return (
      <span className="flex items-center gap-2 font-mono text-xs">
        <span style={{ color: r.is_valid ? '#00cc66' : '#ff6666' }}>{r.is_valid ? '✓ valid' : '⚠ violation'}</span>
        <span className="text-white/30">{(r.completion ?? []).length} steps</span>
      </span>
    )
  }
  return (
    <span className="flex items-center gap-2 font-mono text-xs flex-wrap">
      <span
        className="font-bold"
        style={{ color: r.is_valid ? '#00cc66' : '#ff5555' }}
      >
        {r.is_valid ? '✓ VALID' : '⚠ ANOMALY'}
      </span>
      {!r.is_valid && r.predicted_rule && (
        <span
          className="px-1.5 py-0.5 rounded text-xs"
          style={{ backgroundColor: '#ff444418', color: '#ff7777', border: '1px solid #ff444430' }}
        >
          {r.predicted_rule}
        </span>
      )}
    </span>
  )
}

const CONCURRENCY = 8

export default function InferencePage() {
  const [rows, setRows] = useState<CSVRow[]>([])
  const [task, setTask] = useState<Task>('nextstep')
  const [results, setResults] = useState<BatchResult[]>([])
  const [running, setRunning] = useState(false)
  const [done, setDone] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const [parseError, setParseError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef(false)

  const completed = results.filter((r) => r.status === 'done' || r.status === 'error').length
  const progress = rows.length > 0 ? completed / rows.length : 0

  function loadFile(file: File) {
    const reader = new FileReader()
    reader.onload = (ev) => {
      const { rows: parsed, error } = parseCSV(ev.target?.result as string)
      if (error || parsed.length === 0) {
        setParseError(error ?? 'No rows found.')
        return
      }
      setParseError(null)
      setRows(parsed)
      setResults(parsed.map((row) => ({ row, status: 'pending', response: null })))
      setDone(false)
      setTask(parsed[0].colType === 'SEQUENCE' ? 'anomaly' : 'nextstep')
    }
    reader.readAsText(file)
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) loadFile(file)
    e.target.value = ''
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) loadFile(file)
  }

  async function runBatch() {
    if (rows.length === 0 || running) return
    abortRef.current = false
    setRunning(true)
    setDone(false)

    // Reset all to pending
    const initial: BatchResult[] = rows.map((row) => ({ row, status: 'pending', response: null }))
    setResults([...initial])

    let idx = 0
    const total = rows.length

    async function worker() {
      while (idx < total && !abortRef.current) {
        const i = idx++
        setResults((prev) => {
          const next = [...prev]
          next[i] = { ...next[i], status: 'running' }
          return next
        })
        try {
          let res: LiveInferResponse
          const { steps, family } = rows[i]
          if (task === 'nextstep') res = await postInferNextStep(steps, family)
          else if (task === 'complete') res = await postInferComplete(steps, family)
          else res = await postInferAnomaly(steps, family)
          setResults((prev) => {
            const next = [...prev]
            next[i] = { ...next[i], status: 'done', response: res }
            return next
          })
        } catch {
          setResults((prev) => {
            const next = [...prev]
            next[i] = { ...next[i], status: 'error', response: null }
            return next
          })
        }
      }
    }

    await Promise.all(Array.from({ length: CONCURRENCY }, worker))
    setRunning(false)
    setDone(true)
  }

  function reset() {
    abortRef.current = true
    setRows([])
    setResults([])
    setRunning(false)
    setDone(false)
    setParseError(null)
  }

  const colType = rows[0]?.colType

  return (
    <div className="max-w-5xl mx-auto space-y-5 py-2">

      <div className="flex items-center justify-between">
        <h1 className="text-white font-bold text-2xl">Inference</h1>
        {rows.length > 0 && (
          <button onClick={reset} className="text-white/25 hover:text-white/50 text-xs font-mono transition-colors">
            ✕ clear
          </button>
        )}
      </div>

      {/* Upload zone — only shown when no file loaded */}
      {rows.length === 0 && (
        <>
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
            className="rounded-xl border-2 border-dashed py-16 text-center cursor-pointer transition-all"
            style={{
              borderColor: parseError ? '#ff444450' : isDragOver ? '#00d4ff50' : '#ffffff15',
              backgroundColor: parseError ? '#ff444408' : isDragOver ? '#00d4ff06' : 'transparent',
            }}
          >
            <div className="font-mono text-sm" style={{ color: parseError ? '#ff7777' : isDragOver ? '#00d4ff' : '#ffffff50' }}>
              {isDragOver ? 'Drop to upload' : 'Drop a CSV file here, or click to upload'}
            </div>
            {parseError && (
              <div className="mt-3 text-xs font-mono px-6" style={{ color: '#ff666660' }}>
                {parseError}
              </div>
            )}
          </div>
          {!parseError && (
            <div className="text-white/20 font-mono text-xs">
              Accepts: eval_input_valid.csv · eval_input_anomaly.csv · MOSFET_variants.csv · syntheticIC.csv · completion.csv
            </div>
          )}
        </>
      )}
      <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={onFileChange} />

      {/* Controls — shown after file loaded */}
      {rows.length > 0 && (
        <div className="flex items-center gap-4 flex-wrap">
          <span className="text-white/50 font-mono text-sm">{rows.length} rows</span>

          {/* Task selector — only for PARTIAL_SEQUENCE */}
          {colType === 'PARTIAL_SEQUENCE' && !running && !done && (
            <div className="flex gap-1">
              {(['nextstep', 'complete'] as Task[]).map((t) => (
                <button
                  key={t}
                  onClick={() => setTask(t)}
                  className={`text-xs font-mono px-3 py-1 rounded transition-colors ${
                    task === t
                      ? 'bg-accent/20 text-accent border border-accent/30'
                      : 'text-white/40 hover:text-white hover:bg-white/5 border border-transparent'
                  }`}
                >
                  {t === 'nextstep' ? 'Task 1 — Next Step' : 'Task 2 — Complete'}
                </button>
              ))}
            </div>
          )}

          {colType === 'SEQUENCE' && (
            <span className="text-xs font-mono text-white/40 border border-white/10 px-2 py-1 rounded">Task 3 — Anomaly</span>
          )}

          {!running && !done && (
            <button
              onClick={runBatch}
              className="px-4 py-1.5 rounded-lg text-sm font-mono font-bold transition-all"
              style={{ backgroundColor: '#00cc66', color: '#000' }}
            >
              ▶ Run all
            </button>
          )}

          {running && (
            <div className="flex items-center gap-3">
              <div className="w-40 h-1.5 rounded-full bg-white/10 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{ width: `${progress * 100}%`, backgroundColor: '#00cc66' }}
                />
              </div>
              <span className="text-white/40 font-mono text-xs tabular-nums">{completed}/{rows.length}</span>
            </div>
          )}

          {done && (
            <button
              onClick={() => exportCSV(results, task)}
              className="text-xs font-mono px-3 py-1 rounded border border-white/15 text-white/50 hover:text-white hover:border-white/30 transition-colors"
            >
              ↓ Export CSV
            </button>
          )}
        </div>
      )}

      {/* Results table */}
      {results.length > 0 && (
        <div className="rounded-xl border overflow-hidden" style={{ borderColor: '#ffffff10' }}>
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b" style={{ backgroundColor: '#060b18', borderColor: '#ffffff10' }}>
                <th className="text-left px-4 py-2.5 text-white/30 font-normal w-40">ID</th>
                <th className="text-left px-3 py-2.5 text-white/30 font-normal w-20">Family</th>
                <th className="text-left px-3 py-2.5 text-white/30 font-normal">Result</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result, i) => (
                <tr
                  key={result.row.exampleId}
                  className="border-b border-white/5 last:border-0"
                  style={{
                    backgroundColor:
                      result.status === 'done' && task === 'anomaly' && result.response?.is_valid === false
                        ? '#ff444408'
                        : result.status === 'done' && task === 'anomaly' && result.response?.is_valid === true
                        ? '#00cc6605'
                        : i % 2 === 0 ? 'transparent' : '#ffffff02',
                  }}
                >
                  <td className="px-4 py-2 text-white/50">{result.row.exampleId}</td>
                  <td className="px-3 py-2 text-white/35">{result.row.family}</td>
                  <td className="px-3 py-2">
                    <ResultCell result={result} task={task} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

    </div>
  )
}
