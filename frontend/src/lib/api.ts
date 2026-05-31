import type { EvalMetrics, CompletionListResponse, CompletionDetail, AnomalyListResponse, AnomalyDetail, LiveInferResponse } from '../types/api'

const BASE_URL = import.meta.env.VITE_API_URL ?? ''

async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = 5000): Promise<Response> {
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(url, { ...options, signal: controller.signal })
    clearTimeout(id)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return response
  } catch (err) {
    clearTimeout(id)
    throw err
  }
}

export async function postPredict(payload: unknown) {
  const res = await fetchWithTimeout(`${BASE_URL}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return res.json()
}

export async function postOptimize(payload: unknown) {
  const res = await fetchWithTimeout(`${BASE_URL}/api/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return res.json()
}

export async function getBatches(params: { page?: number; page_size?: number; sort_by?: string; order?: string }) {
  const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)])).toString()
  const res = await fetchWithTimeout(`${BASE_URL}/api/batches${qs ? '?' + qs : ''}`)
  return res.json()
}

export async function getBatchDetail(batchId: string) {
  const res = await fetchWithTimeout(`${BASE_URL}/api/batches/${batchId}`)
  return res.json()
}

export async function getEvalMetrics(): Promise<EvalMetrics> {
  const res = await fetchWithTimeout(`${BASE_URL}/api/eval/metrics`, {}, 10000)
  return res.json()
}

export async function getCompletions(params: {
  page?: number
  page_size?: number
  family?: string
  sort_by?: string
  order?: string
}): Promise<CompletionListResponse> {
  const qs = new URLSearchParams(
    Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== '')
      .map(([k, v]) => [k, String(v)])
  ).toString()
  const res = await fetchWithTimeout(`${BASE_URL}/api/eval/completions${qs ? '?' + qs : ''}`, {}, 10000)
  return res.json()
}

export async function getCompletionDetail(exampleId: string): Promise<CompletionDetail> {
  const res = await fetchWithTimeout(`${BASE_URL}/api/eval/completions/${encodeURIComponent(exampleId)}`, {}, 10000)
  return res.json()
}

export async function getAnomalies(params: {
  page?: number
  page_size?: number
  family?: string
  is_valid?: string
  sort_by?: string
  order?: string
}): Promise<AnomalyListResponse> {
  const qs = new URLSearchParams(
    Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== '')
      .map(([k, v]) => [k, String(v)])
  ).toString()
  const res = await fetchWithTimeout(`${BASE_URL}/api/eval/anomalies${qs ? '?' + qs : ''}`, {}, 10000)
  return res.json()
}

export async function getAnomalyDetail(exampleId: string): Promise<AnomalyDetail> {
  const res = await fetchWithTimeout(`${BASE_URL}/api/eval/anomalies/${encodeURIComponent(exampleId)}`, {}, 10000)
  return res.json()
}

export async function postInferNextStep(steps: string[], family: string): Promise<LiveInferResponse> {
  const res = await fetchWithTimeout(
    `${BASE_URL}/api/infer/nextstep`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ steps, family }) },
    15000,
  )
  return res.json()
}

export async function postInferComplete(steps: string[], family: string): Promise<LiveInferResponse> {
  const res = await fetchWithTimeout(
    `${BASE_URL}/api/infer/complete`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ steps, family }) },
    15000,
  )
  return res.json()
}

export async function postInferAnomaly(steps: string[], family: string): Promise<LiveInferResponse> {
  const res = await fetchWithTimeout(
    `${BASE_URL}/api/infer/anomaly`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ steps, family }) },
    15000,
  )
  return res.json()
}
