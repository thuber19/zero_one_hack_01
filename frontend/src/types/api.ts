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

export interface EvalMetrics {
  task1_nextstep: { n: number; top1: number; top3: number; top5: number; mrr: number; top1_category: number }
  task2_completion: { n: number; exact_match: number; block_accuracy: number; token_accuracy: number; normalized_edit_distance: number }
  task2_logic_validity: number
  task3_encoder: { n: number; binary_accuracy: number; precision: number; recall: number; f1: number; roc_auc: number }
  task3_hybrid: { n: number; binary_accuracy: number; precision: number; recall: number; f1: number; rule_attribution_accuracy: number }
  task1_hybrid: { n: number; top1: number; top3: number; top5: number; mrr: number; top1_category: number }
}

export interface CompletionRow {
  example_id: string
  family: string
  completion_fraction: number
  partial_step_count: number
  predicted_step_count: number
  rank1: string
  rank2: string
  rank3: string
  rank4: string
  rank5: string
}

export interface CompletionDetail extends CompletionRow {
  partial_steps: string[]
  predicted_sequence_steps: string[]
}

export interface CompletionListResponse {
  total: number
  page: number
  page_size: number
  items: CompletionRow[]
}

export interface AnomalyRow {
  example_id: string
  family: string
  is_valid: boolean
  score: number
  predicted_rule: string
  sequence_step_count: number
}

export interface AnomalyDetail extends AnomalyRow {
  full_sequence: string[]
}

export interface AnomalyListResponse {
  total: number
  page: number
  page_size: number
  items: AnomalyRow[]
}

export interface LiveInferResponse {
  predictions?: string[]
  completion?: string[]
  is_valid?: boolean
  score?: number
  predicted_rule?: string
  available?: boolean
}
