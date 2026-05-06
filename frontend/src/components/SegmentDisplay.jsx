import React, { useState } from 'react';

// RelabelButton for segments (updated with dropdown)
function RelabelButton({ segmentIndex, onConfirm, currentLabel }) {
  const [show, setShow] = useState(false);
  const [mode, setMode] = useState('select'); // select, custom
  const [customLabel, setCustomLabel] = useState('');

  const handleSelect = (e) => {
    const val = e.target.value;
    if (val === 'custom') {
      setMode('custom');
      setCustomLabel('');
    } else if (val) {
      onConfirm(val);
      setShow(false);
    }
  };

  const handleCustomSubmit = () => {
    if (customLabel.trim()) {
      onConfirm(customLabel.trim());
      setShow(false);
      setMode('select');
    }
  };

  return show ? (
    <span style={{ marginLeft: 8, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      {mode === 'select' ? (
        <select onChange={handleSelect} style={{ padding: 2 }} defaultValue="">
          <option value="" disabled>Select Label...</option>
          <option value="work">Work</option>
          <option value="idle">Idle</option>
          <option value="custom">Custom...</option>
        </select>
      ) : (
        <>
          <input
            value={customLabel}
            onChange={e => setCustomLabel(e.target.value)}
            placeholder="Label"
            autoFocus
            style={{ width: 80, padding: 2 }}
          />
          <button onClick={handleCustomSubmit} style={{ cursor: 'pointer' }}>OK</button>
        </>
      )}
      <button onClick={() => { setShow(false); setMode('select'); }} style={{ cursor: 'pointer', fontSize: '0.85em' }}>Cancel</button>
    </span>
  ) : (
    <button
      style={{ marginLeft: 8, color: 'orange', borderColor: 'orange', background: 'transparent', border: '1px solid orange', borderRadius: 4, cursor: 'pointer', fontSize: '0.85em', padding: '1px 6px' }}
      onClick={() => setShow(true)}
    >
      Relabel
    </button>
  );
}

export default function SegmentDisplay({ segments = [], lowConfidenceFrames = [], onRelabel }) {
  const [selectedIndices, setSelectedIndices] = useState([]);

  if (!segments || segments.length === 0) {
    return (
      <div style={{ padding: 12, border: '1px dashed var(--panel-border)', borderRadius: 8, color: 'var(--muted)' }}>
        No temporal segments yet (requires classifier_source=llm)
      </div>
    );
  }

  // Pre-calculate low confidence status for each segment
  const segmentStatus = segments.map((seg) => {
    const isLowConf = lowConfidenceFrames && seg.start_time !== undefined && seg.end_time !== undefined &&
      lowConfidenceFrames.some(fidx => {
        const fps = seg.fps || 30;
        const t = fidx / fps;
        return t >= seg.start_time && t <= seg.end_time;
      });
    return { isLowConf };
  });

  const toggleSelect = (idx) => {
    setSelectedIndices(prev =>
      prev.includes(idx) ? prev.filter(i => i !== idx) : [...prev, idx]
    );
  };

  const selectAllLowConf = () => {
    const lowConfIndices = segments.map((_, idx) => idx).filter(idx => segmentStatus[idx].isLowConf);
    setSelectedIndices(lowConfIndices);
  };

  const handleBulkRelabel = (newLabel) => {
    if (!newLabel) return;
    if (confirm(`Relabel ${selectedIndices.length} segments as "${newLabel}"?`)) {
      // Iterate through selected segments and call onRelabel for each
      // Note: This might trigger multiple re-renders or API calls. 
      // In a production app, we'd want a bulk API endpoint.
      selectedIndices.forEach(idx => {
        if (segments[idx]) {
          onRelabel(segments[idx], newLabel);
        }
      });
      setSelectedIndices([]);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>

      {/* Bulk Actions Bar */}
      <div style={{ padding: 8, background: 'var(--card-bg)', border: '1px solid var(--panel-border)', borderRadius: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <button onClick={selectAllLowConf} style={{ fontSize: '0.9em', marginRight: 10 }}>Select Low Confidence</button>
          <span style={{ fontSize: '0.9em', color: 'var(--muted)' }}>
            {selectedIndices.length} selected
          </span>
        </div>

        {selectedIndices.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: '0.9em' }}>Relabel selected as:</span>
            <button onClick={() => handleBulkRelabel('work')} style={{ background: '#28a745', color: 'white', border: 'none', borderRadius: 4, padding: '2px 8px', cursor: 'pointer' }}>Work</button>
            <button onClick={() => handleBulkRelabel('idle')} style={{ background: '#6c757d', color: 'white', border: 'none', borderRadius: 4, padding: '2px 8px', cursor: 'pointer' }}>Idle</button>
            <button onClick={() => {
              const l = prompt("Enter custom label:");
              if (l) handleBulkRelabel(l);
            }} style={{ cursor: 'pointer' }}>Custom</button>
          </div>
        )}
      </div>

      {segments.map((seg, idx) => {
        const { isLowConf } = segmentStatus[idx];
        const isSelected = selectedIndices.includes(idx);

        return (
          <div key={idx} style={{
            padding: 12,
            border: isSelected ? '2px solid var(--accent)' : (isLowConf ? '2px solid orange' : '1px solid var(--panel-border)'),
            borderRadius: 8,
            backgroundColor: isSelected ? 'rgba(var(--accent-rgb), 0.05)' : (isLowConf ? 'rgba(255, 200, 0, 0.05)' : 'var(--panel-bg)'),
            transition: 'all 0.2s'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => toggleSelect(idx)}
                />
                <span>Segment {idx + 1}</span>
                <span style={{ fontWeight: 400, color: 'var(--muted)' }}>
                  {seg.start_time?.toFixed(2)}s – {seg.end_time?.toFixed(2)}s
                </span>
              </div>
              <div style={{ color: 'var(--muted)' }}>
                Duration: {seg.duration?.toFixed(2)}s
              </div>
            </div>

            <div style={{ marginBottom: 6, paddingLeft: 24 }}>
              <span style={{ fontWeight: 600 }}>Dominant Caption:</span>{' '}
              <span>{seg.dominant_caption || '—'}</span>
            </div>

            {seg.label && (
              <div style={{ marginBottom: 6, display: 'flex', alignItems: 'center', paddingLeft: 24 }}>
                <span style={{ fontWeight: 600 }}>Label:</span>{' '}
                <span style={{
                  padding: '2px 8px',
                  borderRadius: 4,
                  backgroundColor: seg.label === 'work' ? '#28a745' : '#6c757d',
                  color: '#fff',
                  fontSize: '0.9em',
                  marginLeft: 6
                }}>
                  {seg.label}
                </span>
                {isLowConf && <span style={{ color: 'orange', fontWeight: 600, marginLeft: 8, fontSize: '0.9em' }}>Low confidence</span>}

                <RelabelButton
                  segmentIndex={idx}
                  currentLabel={seg.label}
                  onConfirm={(lbl) => onRelabel && onRelabel(seg, lbl)}
                />
              </div>
            )}

            {seg.llm_output && (
              <div style={{ marginTop: 6, fontSize: '0.9em', paddingLeft: 24 }}>
                <span style={{ fontWeight: 600 }}>LLM Output:</span>{' '}
                <span style={{ color: 'var(--muted)' }}>{seg.llm_output}</span>
              </div>
            )}

            {seg.prompt && (
              <details style={{ marginTop: 8, paddingLeft: 24 }}>
                <summary style={{ cursor: 'pointer', fontSize: '0.9em', color: 'var(--muted)' }}>
                  View LLM Prompt
                </summary>
                <pre style={{
                  marginTop: 6,
                  padding: 8,
                  backgroundColor: 'var(--code-bg)',
                  borderRadius: 4,
                  fontSize: '0.85em',
                  maxHeight: 200,
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word'
                }}>
                  {seg.prompt}
                </pre>
              </details>
            )}

            {seg.timeline && (
              <details style={{ marginTop: 8, paddingLeft: 24 }}>
                <summary style={{ cursor: 'pointer', fontSize: '0.9em', color: 'var(--muted)' }}>
                  View Timeline ({seg.captions?.length || 0} samples)
                </summary>
                <pre style={{
                  marginTop: 6,
                  padding: 8,
                  backgroundColor: 'var(--code-bg)',
                  borderRadius: 4,
                  fontSize: '0.85em',
                  maxHeight: 200,
                  overflow: 'auto'
                }}>
                  {seg.timeline}
                </pre>
              </details>
            )}
          </div>
        );
      })}
    </div>
  );
}
