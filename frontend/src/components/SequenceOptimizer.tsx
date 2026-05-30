import { useState, useEffect } from 'react'
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useOptimize } from '../hooks/useOptimize'
import type { Step } from '../types/api'

interface SortableStepProps {
  step: Step
  canonicalIndex: number
  currentIndex: number
}

function SortableStep({ step, canonicalIndex, currentIndex }: SortableStepProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: step.step_id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  }

  const isOutOfOrder = canonicalIndex !== currentIndex && Math.abs(canonicalIndex - currentIndex) > 0

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className="flex items-center gap-3 px-3 py-2 bg-[#0d1426] border border-white/10 rounded cursor-grab active:cursor-grabbing hover:border-accent/40 transition-colors"
    >
      <span className="text-white/20 font-mono text-xs w-8">{String(currentIndex + 1).padStart(2, '0')}</span>
      <div className="flex-1">
        <span className="font-mono text-sm text-white/80">{step.step_name}</span>
        <span className="ml-2 text-xs text-white/30 font-mono">{step.category}</span>
      </div>
      {isOutOfOrder && (
        <span className="text-xs font-mono px-2 py-0.5 bg-risk-amber/10 border border-risk-amber/30 text-risk-amber rounded">
          Non-standard order
        </span>
      )}
      <span
        className={`text-xs font-mono w-16 text-right ${
          step.risk_score >= 0.85 ? 'text-risk-red' : step.risk_score >= 0.70 ? 'text-risk-amber' : 'text-white/30'
        }`}
      >
        {step.risk_score.toFixed(3)}
      </span>
    </div>
  )
}

interface Props {
  steps: Step[]
}

export default function SequenceOptimizer({ steps: initialSteps }: Props) {
  const [steps, setSteps] = useState(initialSteps)
  const [canonicalOrder] = useState(() => initialSteps.map((s) => s.step_id))
  const { optimize, predictedYield, isLoading } = useOptimize()

  useEffect(() => {
    setSteps(initialSteps)
  }, [initialSteps])

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  )

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event
    if (!over || active.id === over.id) return

    setSteps((prev) => {
      const oldIdx = prev.findIndex((s) => s.step_id === active.id)
      const newIdx = prev.findIndex((s) => s.step_id === over.id)
      const reordered = arrayMove(prev, oldIdx, newIdx)
      optimize(reordered.map((s) => ({ step_id: s.step_id, step_name: s.step_name, category: s.category })))
      return reordered
    })
  }

  const displayYield = predictedYield !== null ? predictedYield : (initialSteps.length > 0 ? 0.614 : 0)

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-white font-mono text-base">Step Sequence Optimizer</h2>
          <p className="text-white/40 text-xs font-mono mt-0.5">Drag steps to reorder — yield updates automatically</p>
        </div>
        <div className="text-right">
          <div className={`font-mono text-2xl font-bold tabular-nums ${
            displayYield >= 0.85 ? 'text-yield-green' : displayYield >= 0.70 ? 'text-risk-amber' : 'text-risk-red'
          }`}>
            {isLoading ? (
              <span className="text-white/40 text-lg">⟳</span>
            ) : (
              `${(displayYield * 100).toFixed(1)}%`
            )}
          </div>
          <div className="text-white/40 text-xs font-mono">Predicted Yield</div>
        </div>
      </div>

      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
        <SortableContext items={steps.map((s) => s.step_id)} strategy={verticalListSortingStrategy}>
          <div className="space-y-1 max-h-[60vh] overflow-y-auto pr-1">
            {steps.map((step, idx) => (
              <SortableStep
                key={step.step_id}
                step={step}
                canonicalIndex={canonicalOrder.indexOf(step.step_id)}
                currentIndex={idx}
              />
            ))}
          </div>
        </SortableContext>
      </DndContext>
    </div>
  )
}
