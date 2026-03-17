

import { useState, useRef, useEffect, useMemo } from 'react'

// AA color by physicochemical property
const AA_COLORS = {
  // Hydrophobic
  A: '#4a90d9', V: '#4a90d9', I: '#4a90d9', L: '#4a90d9',
  M: '#5ba3e8', F: '#7b68ee', W: '#9370db', P: '#4169e1',
  // Polar uncharged
  G: '#98d982', S: '#66cc66', T: '#55bb55', C: '#e8d44d',
  Y: '#d4a843', N: '#66d9b8', Q: '#55c9a8',
  // Charged positive
  K: '#ff6b6b', R: '#ff4444', H: '#ff9966',
  // Charged negative
  D: '#ff6b9d', E: '#ff4488',
}

const EXAMPLE_SEQUENCES = [
  {
    label: 'Villin HP35 (35aa)',
    seq: 'LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
  },
  {
    label: 'Trp-cage (20aa)',
    seq: 'NLYIQWLKDGGPSSGRPPPS',
  },
  {
    label: 'GB1 domain (56aa)',
    seq: 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE',
  },
]

export default function SequencePanel({
  onSubmit,
  status,
  progress,
  message,
  residues,
  attentionData,
  selectedResidue,
  onResidueSelect,
}) {
  const [sequence, setSequence] = useState('')
  const [isValid, setIsValid] = useState(false)
  const heatmapRef = useRef(null)

  const isDone = status === 'DONE'
  const isRunning = status === 'RUNNING' || status === 'PENDING'

  const validate = (seq) => {
    const cleaned = seq.trim().toUpperCase()
    const valid = /^[ACDEFGHIKLMNPQRSTVWY]{10,400}$/.test(cleaned)
    setIsValid(valid)
    return valid
  }

  const handleChange = (e) => {
    const v = e.target.value.replace(/\s/g, '').toUpperCase()
    setSequence(v)
    validate(v)
  }

  const handleSubmit = () => {
    if (isValid && !isRunning) {
      onSubmit(sequence.trim().toUpperCase())
    }
  }

  // Residue score lookup
  const scores = attentionData?.residue_scores || []

  const getColor = (aa, index) => {
    const base = AA_COLORS[aa] || '#666'
    if (!scores.length || index >= scores.length) return base
    const s = scores[index]  // 0-1
    // Interpolate from base toward hot red
    return `rgba(${Math.round(30 + s * 225)}, ${Math.round(60 + s * 20)}, ${Math.round(200 - s * 180)}, 0.9)`
  }

  // Draw mini heatmap
  useEffect(() => {
    if (!heatmapRef.current || !attentionData?.heatmap) return

    const canvas = heatmapRef.current
    const { shape, data } = attentionData.heatmap
    const N = shape[0]
    const ctx = canvas.getContext('2d')

    canvas.width = N
    canvas.height = N

    const imgData = ctx.createImageData(N, N)
    for (let i = 0; i < N * N; i++) {
      const v = data[i]
      // Cool to hot colormap
      const r = Math.round(Math.min(255, v * 3 * 255))
      const g = Math.round(Math.min(255, Math.max(0, (v - 0.33) * 3 * 255)))
      const b = Math.round(Math.max(0, (1 - v * 2) * 255))
      imgData.data[i * 4]     = r
      imgData.data[i * 4 + 1] = g
      imgData.data[i * 4 + 2] = b
      imgData.data[i * 4 + 3] = 255
    }
    ctx.putImageData(imgData, 0, 0)
  }, [attentionData])

  const selectedInfo = useMemo(() => {
    if (selectedResidue === null || !residues.length) return null
    const r = residues[selectedResidue]
    if (!r) return null
    const connections = (attentionData?.edges || []).filter(
      e => e.i === selectedResidue || e.j === selectedResidue
    )
    return { residue: r, connections }
  }, [selectedResidue, residues, attentionData])

  return (
    <>
      {/* Input */}
      <div className="card">
        <div className="card-title"><span className="dot" />Sequence Input</div>
        <textarea
          value={sequence}
          onChange={handleChange}
          placeholder="Enter amino acid sequence (10–400 residues)&#10;e.g. MKTAYIAKQRQ..."
          style={{ marginBottom: 10, fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}
        />

        {/* Examples */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 10 }}>
          {EXAMPLE_SEQUENCES.map(({ label, seq }) => (
            <button
              key={label}
              className="btn btn-ghost btn-sm"
              onClick={() => {
                setSequence(seq)
                validate(seq)
              }}
            >
              {label}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            {sequence.length} aa
            {!isValid && sequence.length > 0 && (
              <span style={{ color: 'var(--accent-pink)', marginLeft: 6 }}>invalid</span>
            )}
          </span>
          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={!isValid || isRunning}
          >
            {isRunning ? '⏳ Running…' : '▶ Predict'}
          </button>
        </div>
      </div>

      {/* Status */}
      {status && (
        <div className="card">
          <div className="card-title"><span className="dot" />Status</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <span className={`status-badge ${status.toLowerCase()}`}>
              <span className="pulse" />
              {status}
            </span>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{message}</span>
          </div>
          {isRunning && (
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
          )}
        </div>
      )}

      {/* Sequence strip */}
      {residues.length > 0 && (
        <div className="card">
          <div className="card-title"><span className="dot" />Sequence ({residues.length} aa)</div>
          <div className="sequence-strip">
            {residues.map((r, i) => (
              <div
                key={i}
                className={`residue-chip ${selectedResidue === i ? 'selected' : ''}`}
                style={{ background: getColor(r.aa, i), color: '#000' }}
                onClick={() => onResidueSelect(i)}
                title={`${r.aa}${r.resno} — score: ${scores[i]?.toFixed(3) || 'n/a'}`}
              >
                {r.aa}
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8 }}>
            <div className="legend">
              <span className="legend-label">Low attn</span>
              <div className="legend-gradient" />
              <span className="legend-label">High attn</span>
            </div>
          </div>
        </div>
      )}

      {/* Selected residue info */}
      {selectedInfo && (
        <div className="card" style={{ borderColor: 'var(--accent-blue)44', boxShadow: 'var(--glow-blue)' }}>
          <div className="card-title">
            <span className="dot" style={{ background: 'var(--accent-gold)' }} />
            Residue {selectedInfo.residue.aa}{selectedInfo.residue.resno}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12 }}>
            {[
              ['AA', selectedInfo.residue.aa],
              ['Index', selectedInfo.residue.index],
              ['Chain', selectedInfo.residue.chain],
              ['pLDDT', selectedInfo.residue.plddt?.toFixed(1) || '—'],
              ['Attention score', scores[selectedInfo.residue.index]?.toFixed(4) || '—'],
              ['Connections', selectedInfo.connections.length],
            ].map(([label, val]) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{val}</span>
              </div>
            ))}

            {selectedInfo.connections.length > 0 && (
              <>
                <div className="sep" />
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                  Top connections:
                </div>
                {selectedInfo.connections.slice(0, 5).map((e, i) => {
                  const partner = e.i === selectedInfo.residue.index ? e.j : e.i
                  const pr = residues[partner]
                  return pr ? (
                    <div
                      key={i}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        cursor: 'pointer',
                        padding: '2px 4px',
                        borderRadius: 4,
                        fontSize: 11,
                      }}
                      onClick={() => onResidueSelect(partner)}
                    >
                      <span style={{ color: 'var(--text-secondary)' }}>
                        → {pr.aa}{pr.resno} (sep {Math.abs(partner - selectedInfo.residue.index)})
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                        {e.weight.toFixed(3)}
                      </span>
                    </div>
                  ) : null
                })}
              </>
            )}
          </div>
        </div>
      )}

      {/* Mini heatmap */}
      {attentionData?.heatmap && (
        <div className="card">
          <div className="card-title">
            <span className="dot" style={{ background: 'var(--accent-pink)' }} />
            Attention Matrix (Layer {(attentionData.layer ?? 0) + 1})
          </div>
          <div className="heatmap-container">
            <canvas ref={heatmapRef} style={{ imageRendering: 'pixelated' }} />
          </div>
          <div className="legend" style={{ marginTop: 8 }}>
            <span className="legend-label">0</span>
            <div className="legend-gradient" />
            <span className="legend-label">1</span>
          </div>
        </div>
      )}
    </>
  )
}
