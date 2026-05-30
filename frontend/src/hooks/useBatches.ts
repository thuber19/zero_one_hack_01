import { useQuery } from '@tanstack/react-query'
import { getBatches } from '../lib/api'
import type { BatchListResponse } from '../types/api'

// Fallback fixture data for when API is unavailable
export const FALLBACK_BATCHES: BatchListResponse = {
  total: 10,
  page: 1,
  page_size: 20,
  batches: Array.from({ length: 10 }, (_, i) => ({
    batch_id: `batch_${String(i + 1).padStart(3, '0')}`,
    timestamp: new Date(Date.now() - i * 3600000).toISOString(),
    material: i % 3 === 0 ? 'IGBT' : i % 3 === 1 ? 'MOSFET' : 'IC',
    predicted_yield: parseFloat((0.5 + ((i * 7) % 5) * 0.1).toFixed(2)),
    defect_probability: parseFloat((0.9 - ((i * 7) % 5) * 0.1).toFixed(2)),
    confidence: parseFloat((0.7 + (i % 3) * 0.1).toFixed(2)),
    risk_steps_detected: (i * 3) % 20,
  })),
}

interface UseBatchesParams {
  page?: number
  pageSize?: number
  sortBy?: string
  order?: 'asc' | 'desc'
}

export function useBatches({
  page = 1,
  pageSize = 20,
  sortBy = 'defect_probability',
  order = 'desc',
}: UseBatchesParams = {}) {
  return useQuery<BatchListResponse>({
    queryKey: ['batches', page, pageSize, sortBy, order],
    queryFn: () => getBatches({ page, page_size: pageSize, sort_by: sortBy, order }),
    placeholderData: (prev) => prev,
    select: (data) => data,
    retry: 1,
  })
}
