/**
 * AttentionControls — right panel
 *
 * - Layer slider
 * - Head aggregation selector
 * - Threshold slider
 * - Animation toggle
 * - Layer profile bar chart
 * - Color mode selector
 * - Edge count info
 */

import { useState, useEffect, useRef, useCallback } from 'react'

const COLOR_MODES = [
  { value: 'attention', label: 'Attention' },
  { value: 'bfactor',   label: 'pLDDT' },
  { value: 'chainid',   label: 'Chain' },
  { value: 'resname',   label: 'Residue type' },
]

export default function AttentionControls({
  numLayers = 0,
  attentionData,
  layerProfile,
  layer,
  threshold,
  showEdges,
  colorMode,
  isAnimating,
  onLayerChange,
  onThresholdChange,
  onShowEdgesChange,
  onColorModeChange,
  onAnimationToggle,
}) {
  const animRef = useRef(null)

  // Auto-advance layer for animation
  useEffect(() => {
    if (isAnimating && numLayers > 0) {
      animRef.current = setInterval(() => {
        onLayerChange(prev => (prev + 1) % numLayers)
      }, 800)
    } else {
      if (animRef.current) clearInterval(animRef.current)
    }
    return () => { if (animRef.current) clearInterval(animRef.current) }
  }, [isAnimating, numLayers, onLayerChange])

  const profileLayers = layerProfile?.layers || []
  const maxEntropy = Math.max(...profileLayers.map(l => l.entropy), 1)

  return (
    <>
      {/* Layer selection */}
      <div className="card">
        <div className="card-title"><span className="dot" />Attention Layer</div>

        <div className="slider-group">
          <div className="slider-header">
            <span className="slider-label">Layer</span>
            <span className="slider-value">{layer + 1} / {numLayers || '—'}</span>
          </div>
          <input
            type="range"
            min={0}
            max={Math.max(0, numLayers - 1)}
            value={layer}
            onChange={e => onLayerChange(Number(e.target.value))}
            disabled={!numLayers}
          />
        </div>

        {/* Layer profile sparkbar */}
        {profileLayers.length > 0 && (
          <div style={{ marginTop: 10 }}>
            <div className="slider-label" style={{ marginBottom: 6, fontSize: 11 }}>
              Layer entropy (higher = more distributed attention)
            </div>
            <div className="layer-bars">
              {profileLayers.map((lp, i) => (
                <div
                  key={i}
                  className={`layer-bar ${i === layer ? 'active' : ''}`}
                  style={{
                    height: `${(lp.entropy / maxEntropy) * 100}%`,
                    background: i === layer
                      ? 'var(--accent-blue)'
                      : `hsl(${180 + (lp.long_range_ratio ?? lp.local_bias ?? 0) * 60}, 80%, 50%)`,
                    opacity: i === layer ? 1 : 0.5,
                  }}
                  onClick={() => onLayerChange(i)}
                  title={`Layer ${i + 1} — entropy: ${lp.entropy?.toFixed(3)}, LR: ${(lp.long_range_ratio ?? lp.local_bias)?.toFixed(2) ?? 'n/a'}`}
                />
              ))}
            </div>
          </div>
        )}

        {/* Animation */}
        <div style={{ marginTop: 12 }}>
          <label className="toggle" onClick={onAnimationToggle}>
            <div className={`toggle-track ${isAnimating ? 'on' : ''}`}>
              <div className="toggle-thumb" />
            </div>
            <span>Animate layers</span>
          </label>
        </div>
      </div>

      {/* Threshold */}
      <div className="card">
        <div className="card-title"><span className="dot" style={{ background: 'var(--accent-cyan)' }} />Attention Edges</div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <label className="toggle" onClick={() => onShowEdgesChange(!showEdges)}>
            <div className={`toggle-track ${showEdges ? 'on' : ''}`}>
              <div className="toggle-thumb" />
            </div>
            <span>Show connections</span>
          </label>
        </div>

        <div className="slider-group">
          <div className="slider-header">
            <span className="slider-label">Threshold</span>
            <span className="slider-value">{threshold.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={threshold}
            onChange={e => onThresholdChange(Number(e.target.value))}
          />
        </div>

        {attentionData && (
          <div className="info-pills" style={{ marginTop: 10 }}>
            <div className="info-pill">
              Edges: <span>{attentionData.edges?.length || 0}</span>
            </div>
            {attentionData.edges?.length > 0 && (
              <div className="info-pill">
                Max: <span>{Math.max(...attentionData.edges.map(e => e.weight)).toFixed(3)}</span>
              </div>
            )}
          </div>
        )}

        {/* Legend */}
        <div className="legend" style={{ marginTop: 12 }}>
          <span className="legend-label">Low</span>
          <div className="legend-gradient" />
          <span className="legend-label">High</span>
        </div>
      </div>

      {/* Color mode */}
      <div className="card">
        <div className="card-title"><span className="dot" style={{ background: 'var(--accent-pink)' }} />Color Mode</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {COLOR_MODES.map(({ value, label }) => (
            <label
              key={value}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                cursor: 'pointer',
                padding: '5px 8px',
                borderRadius: 6,
                fontSize: 12,
                color: colorMode === value ? 'var(--accent-blue)' : 'var(--text-secondary)',
                background: colorMode === value ? 'var(--accent-dim)' : 'transparent',
                transition: 'all 0.15s',
              }}
              onClick={() => onColorModeChange(value)}
            >
              <div style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: colorMode === value ? 'var(--accent-blue)' : 'var(--border)',
                flexShrink: 0,
              }} />
              {label}
            </label>
          ))}
        </div>
      </div>

      {/* Layer stats */}
      {layerProfile?.layers && (
        <div className="card">
          <div className="card-title"><span className="dot" style={{ background: 'var(--accent-gold)' }} />Layer Stats</div>
          <LayerStats profile={layerProfile.layers[layer]} layer={layer} />
        </div>
      )}
    </>
  )
}

function LayerStats({ profile, layer }) {
  if (!profile) return null
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {[
        ['Layer', layer + 1],
        ['Entropy', profile.entropy?.toFixed(4)],
        ['Long-range / local bias', (profile.long_range_ratio ?? profile.local_bias)?.toFixed(3) ?? '—'],
        ['Sparsity', profile.sparsity?.toFixed(3) ?? '—'],
      ].map(([label, value]) => (
        <div key={label} style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 12,
        }}>
          <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{value}</span>
        </div>
      ))}
    </div>
  )
}
