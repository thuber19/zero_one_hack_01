export interface ShapContribution {
  feature: string
  value: number
  contribution: number
}

export interface Anomaly {
  step_id: string
  anomaly_score: number
  type: string
}

export interface Step {
  step_id: string
  step_name: string
  category: string
  risk_score: number
  confidence_lo: number
  confidence_hi: number
  shap: ShapContribution[]
  duration_min?: number
  sensors?: Record<string, number>
}

export interface PredictResponse {
  batch_id: string
  predicted_yield: number
  confidence: number
  risk_steps_detected: number
  anomalous_batches: number
  per_step: Step[]
  anomalies: Anomaly[]
}

export interface BatchSummary {
  batch_id: string
  timestamp: string
  material: string
  predicted_yield: number
  defect_probability: number
  confidence: number
  risk_steps_detected: number
}

export interface BatchListResponse {
  total: number
  page: number
  page_size: number
  batches: BatchSummary[]
}

export interface OptimizeResponse {
  predicted_yield: number
  confidence: number
  cached: boolean
}
