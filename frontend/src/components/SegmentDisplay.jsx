import React from 'react';

export default function SegmentDisplay({ segments = [] }) {
  if (!segments || segments.length === 0) {
    return (
      <div style={{ padding: 12, border: '1px dashed var(--panel-border)', borderRadius: 8, color: 'var(--muted)' }}>
        No temporal segments yet (requires classifier_source=llm)
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {segments.map((seg, idx) => (
        <div key={idx} style={{ 
          padding: 12, 
          border: '1px solid var(--panel-border)', 
          borderRadius: 8, 
          backgroundColor: 'var(--panel-bg)' 
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
            <div style={{ fontWeight: 600 }}>
              Segment {idx + 1}
              <span style={{ marginLeft: 8, fontWeight: 400, color: 'var(--muted)' }}>
                {seg.start_time?.toFixed(2)}s – {seg.end_time?.toFixed(2)}s
              </span>
            </div>
            <div style={{ color: 'var(--muted)' }}>
              Duration: {seg.duration?.toFixed(2)}s
            </div>
          </div>

          <div style={{ marginBottom: 6 }}>
            <span style={{ fontWeight: 600 }}>Dominant Caption:</span>{' '}
            <span>{seg.dominant_caption || '—'}</span>
          </div>

          {seg.label && (
            <div style={{ marginBottom: 6 }}>
              <span style={{ fontWeight: 600 }}>Label:</span>{' '}
              <span style={{ 
                padding: '2px 8px', 
                borderRadius: 4, 
                backgroundColor: seg.label === 'work' ? '#28a745' : '#6c757d',
                color: '#fff',
                fontSize: '0.9em'
              }}>
                {seg.label}
              </span>
            </div>
          )}

          {seg.llm_output && (
            <div style={{ marginTop: 6, fontSize: '0.9em' }}>
              <span style={{ fontWeight: 600 }}>LLM Output:</span>{' '}
              <span style={{ color: 'var(--muted)' }}>{seg.llm_output}</span>
            </div>
          )}

          {seg.prompt && (
            <details style={{ marginTop: 8 }}>
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
            <details style={{ marginTop: 8 }}>
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
      ))}
    </div>
  );
}
