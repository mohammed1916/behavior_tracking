import React, { useState, useRef, useEffect } from 'react';

export default function LiveView({ model: propModel = '', prompt: propPrompt = '', useLLM: propUseLLM = false, selectedSubtask = '' }) {
  const [running, setRunning] = useState(false);
  const imgRef = useRef(null);
  const eventSourceRef = useRef(null);

  // UI options (initialize from props passed by App)
  const [model, setModel] = useState(propModel || '');
  const [useLLM, setUseLLM] = useState(!!propUseLLM);
  // Note: prompt is supplied from parent via `propPrompt`; do not keep a local duplicate state
  const [subtaskId, setSubtaskId] = useState(selectedSubtask || '');
  const [compareTimings, setCompareTimings] = useState(false);
  const [jpegQuality, setJpegQuality] = useState(80);
  const [maxWidth, setMaxWidth] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Live metadata
  const [captionText, setCaptionText] = useState('');
  const [labelText, setLabelText] = useState('');
  const [llmOutput, setLlmOutput] = useState('');
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    // Keep state in sync if parent props change
    setModel(propModel || '');
    setUseLLM(!!propUseLLM);
    setSubtaskId(selectedSubtask || '');
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [propModel, propPrompt, propUseLLM, selectedSubtask]);

  const buildUrl = () => {
    const base = `${window.location.protocol}//${window.location.hostname}:8001`;
    const params = new URLSearchParams();
    if (model) params.set('model', model);
    if (useLLM) params.set('use_llm', 'true');
    // Use the parent-provided prompt (no local duplicate)
    if (propPrompt) params.set('prompt', propPrompt);
    if (subtaskId) params.set('subtask_id', subtaskId);
    if (compareTimings) params.set('compare_timings', 'true');
    if (jpegQuality) params.set('jpeg_quality', String(jpegQuality));
    if (maxWidth) params.set('max_width', String(maxWidth));
    return base + '/backend/stream_pose' + (Array.from(params).length ? ('?' + params.toString()) : '');
  };

  const startLive = () => {
    const url = buildUrl();

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setCaptionText('Initializing...');
    setLabelText('');
    setLlmOutput('');
    setStatusMessage('Connecting...');

    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onopen = () => {
      console.log('Stream connected');
      setRunning(true);
      setStatusMessage('Connected');
    };

    es.onmessage = (ev) => {
      // Try parse JSON structured events
      let data = null;
      try {
        data = JSON.parse(ev.data);
      } catch (e) {
        // ignore
      }

      if (data) {
        if (data.stage) {
          setStatusMessage(data.message || data.stage);
        }
        if (data.caption) setCaptionText(data.caption);
        if (data.label) setLabelText(data.label);
        if (data.llm_output) setLlmOutput(data.llm_output);
        const b64 = data.image || data.frame || data.jpeg || data.jpg;
        if (b64 && imgRef.current) {
          imgRef.current.src = `data:image/jpeg;base64,${b64}`;
        }
      } else {
        // Fallback: some servers might emit raw base64 frames
        if (ev.data && ev.data.length > 50 && imgRef.current) {
          imgRef.current.src = `data:image/jpeg;base64,${ev.data}`;
        }
      }
    };

    es.onerror = (err) => {
      console.warn('Stream error', err);
      setStatusMessage('Stream error');
      // keep state; allow user to stop
    };
  };

  const stopLive = () => {
    setRunning(false);

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    if (imgRef.current) imgRef.current.src = '';

    setStatusMessage('Stopped');

    const base = `${window.location.protocol}//${window.location.hostname}:8001`;
    fetch(base + '/backend/stop_webcam', { method: 'POST' }).catch(() => {});
  };

  return (
    <div className="panel">
      <h4>Live View</h4>
      <div style={{ marginTop: 8 }}>
        <div style={{ color: 'var(--muted)', marginBottom: 8 }}>Connect to Webcam or server stream here.</div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8 }}>
          <div style={{ color: 'var(--muted)' }}><strong>Model:</strong> {model || '(default)'}</div>
          <div style={{ color: 'var(--muted)' }}><strong>Use LLM:</strong> {useLLM ? 'Yes' : 'No'}</div>
          <div style={{ marginLeft: 'auto' }}>
            <button onClick={() => setShowAdvanced(s => !s)}>{showAdvanced ? 'Hide' : 'Advanced'}</button>
          </div>
        </div>

        {showAdvanced && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
            {/* Prompt is managed in App; no local prompt input to avoid duplication */}
            <div>
              <label>Subtask ID (optional):</label>
              <input style={{ width: '100%' }} value={subtaskId} onChange={(e) => setSubtaskId(e.target.value)} placeholder="subtask uuid" />
            </div>
            <div>
              <label>Compare Timings:</label>
              <input type="checkbox" checked={compareTimings} onChange={(e) => setCompareTimings(e.target.checked)} />
            </div>
            <div>
              <label>JPEG Quality: {jpegQuality}</label>
              <input type="range" min={10} max={95} value={jpegQuality} onChange={(e) => setJpegQuality(parseInt(e.target.value || '80'))} />
            </div>
            <div>
              <label>Max Width (px):</label>
              <input style={{ width: '100%' }} value={maxWidth} onChange={(e) => setMaxWidth(e.target.value)} placeholder="e.g. 640" />
            </div>
          </div>
        )}

        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button onClick={running ? stopLive : startLive}>
            {running ? 'Stop' : 'Start'} Live
          </button>
          <div style={{ color: 'var(--muted)' }}>{running ? 'Status: Running...' : 'Status: Not Running'}</div>
          <div style={{ marginLeft: 'auto', color: 'var(--muted)' }}>{statusMessage}</div>
        </div>

        <div style={{ marginTop: 12, display: 'flex', gap: 12 }}>
          <div style={{ flex: 1 }}>
            {running ? (
              <img ref={imgRef} style={{ width: '100%', borderRadius: 8 }} alt="Live stream" />
            ) : (
              <div style={{ padding: 24, border: '1px dashed var(--panel-border)', borderRadius: 8, color: 'var(--muted)' }}>
                Live view is stopped.
              </div>
            )}
          </div>
          <div style={{ width: 320 }}>
            <div style={{ fontWeight: 600 }}>Caption</div>
            <div style={{ padding: 8, minHeight: 80, border: '1px solid var(--panel-border)', borderRadius: 6 }}>{captionText || '—'}</div>
            <div style={{ marginTop: 8 }}>
              <div style={{ fontWeight: 600 }}>Label</div>
              <div style={{ padding: 8 }}>{labelText || '—'}</div>
            </div>
            <div style={{ marginTop: 8 }}>
              <div style={{ fontWeight: 600 }}>LLM Output</div>
              <div style={{ padding: 8, minHeight: 40 }}>{llmOutput || '—'}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
