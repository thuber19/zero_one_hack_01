import { create } from 'zustand'
import type { Step, PredictResponse } from '../types/api'

interface AppStore {
  activeBatch: PredictResponse | null
  selectedStep: Step | null
  isDegradedMode: boolean
  setActiveBatch: (batch: PredictResponse | null) => void
  setSelectedStep: (step: Step | null) => void
  setIsDegradedMode: (v: boolean) => void
  activeSequenceId: string | null
  activeSequenceType: 'completion' | 'anomaly' | null
  setActiveSequence: (id: string, type: 'completion' | 'anomaly') => void
}

export const useAppStore = create<AppStore>((set) => ({
  activeBatch: null,
  selectedStep: null,
  isDegradedMode: false,
  setActiveBatch: (batch) => set({ activeBatch: batch }),
  setSelectedStep: (step) => set({ selectedStep: step }),
  setIsDegradedMode: (v) => set({ isDegradedMode: v }),
  activeSequenceId: null,
  activeSequenceType: null,
  setActiveSequence: (id, type) => set({ activeSequenceId: id, activeSequenceType: type }),
}))
