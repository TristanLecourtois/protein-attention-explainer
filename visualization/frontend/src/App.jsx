/**
 * App — root layout
 *
 * Layout:
 *   [Header]
 *   [SequencePanel] | [ProteinViewer 3D] | [AttentionControls]
 */

import { useState, useCallback, useEffect } from 'react'
import ProteinViewer from './components/ProteinViewer.jsx'
import SequencePanel from './components/SequencePanel.jsx'
import AttentionControls from './components/AttentionControls.jsx'
import { useProteinData } from './hooks/useProteinData.js'

export default function App() {
  const {
    status,
    progress,
    message,
    pdb,
    residues,
    attentionData,
    layerProfile,
    error,
    submit,
    loadAttention,
    loadPdbWithAttention,
    jobId,
  } = useProteinData()

  const [layer, setLayer] = useState(0)
  const [threshold, setThreshold] = useState(0.15)
  const [showEdges, setShowEdges] = useState(true)
  const [colorMode, setColorMode] = useState('attention')
  const [isAnimating, setIsAnimating] = useState(false)
  const [selectedResidue, setSelectedResidue] = useState(null)
  const [viewerPdb, setViewerPdb] = useState(null)

  const numLayers = layerProfile?.num_layers || 0

  // When base PDB loads, show it immediately
  useEffect(() => {
    setViewerPdb(pdb)
  }, [pdb])

  // When layer/colorMode changes, reload attention + PDB with new bfactors
  useEffect(() => {
    if (!jobId || status !== 'DONE') return

    loadAttention(jobId, layer, threshold, 6, 150)

    if (colorMode === 'attention') {
      loadPdbWithAttention(layer).then(p => {
        if (p) setViewerPdb(p)
      })
    }
  }, [layer, jobId, status])

  // Re-fetch edges when threshold changes
  useEffect(() => {
    if (!jobId || status !== 'DONE') return
    loadAttention(jobId, layer, threshold, 6, 150)
  }, [threshold])

  const handleLayerChange = useCallback((val) => {
    const v = typeof val === 'function' ? val(layer) : val
    setLayer(v)
  }, [layer])

  const handleColorModeChange = useCallback((mode) => {
    setColorMode(mode)
    if (mode !== 'attention') {
      setViewerPdb(pdb)  // reset to plain PDB
    } else if (jobId && status === 'DONE') {
      loadPdbWithAttention(layer).then(p => { if (p) setViewerPdb(p) })
    }
  }, [pdb, jobId, status, layer, loadPdbWithAttention])

  const isLoading = status === 'PENDING' || status === 'RUNNING'

  return (
    <div className="app-layout">
      {/* Header */}
      <header className="app-header">
        <span style={{ fontSize: 18 }}>🧬</span>
        <h1>Protein Attention Explainer</h1>
        <span className="subtitle">ESMFold · Attention Maps · 3D Visualization</span>
        <div className="header-spacer" />
        {status && (
          <span className={`status-badge ${status.toLowerCase()}`}>
            <span className="pulse" />
            {status}
          </span>
        )}
        {error && (
          <span style={{ fontSize: 12, color: 'var(--accent-pink)', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            ⚠ {error}
          </span>
        )}
      </header>

      {/* Left sidebar */}
      <aside className="sidebar">
        <SequencePanel
          onSubmit={submit}
          status={status}
          progress={progress}
          message={message}
          residues={residues}
          attentionData={attentionData}
          selectedResidue={selectedResidue}
          onResidueSelect={setSelectedResidue}
        />
      </aside>

      {/* 3D Viewer */}
      <main className="viewer-area">
        <ProteinViewer
          pdb={viewerPdb}
          residues={residues}
          attentionData={attentionData}
          selectedResidue={selectedResidue}
          onResidueClick={setSelectedResidue}
          colorMode={colorMode}
          showEdges={showEdges}
          isLoading={isLoading}
        />

        {/* Viewer info overlay (bottom-left) */}
        {residues.length > 0 && (
          <div style={{
            position: 'absolute',
            bottom: 12,
            left: 12,
            display: 'flex',
            gap: 6,
            flexWrap: 'wrap',
          }}>
            <div className="info-pill">{residues.length} residues</div>
            <div className="info-pill">Layer <span>{layer + 1}/{numLayers}</span></div>
            <div className="info-pill">Edges <span>{attentionData?.edges?.length || 0}</span></div>
            <div className="info-pill">Threshold <span>{threshold.toFixed(2)}</span></div>
          </div>
        )}
      </main>

      {/* Right controls panel */}
      <aside className="controls-panel">
        <AttentionControls
          numLayers={numLayers}
          attentionData={attentionData}
          layerProfile={layerProfile}
          layer={layer}
          threshold={threshold}
          showEdges={showEdges}
          colorMode={colorMode}
          isAnimating={isAnimating}
          onLayerChange={handleLayerChange}
          onThresholdChange={setThreshold}
          onShowEdgesChange={setShowEdges}
          onColorModeChange={handleColorModeChange}
          onAnimationToggle={() => setIsAnimating(a => !a)}
        />
      </aside>
    </div>
  )
}
