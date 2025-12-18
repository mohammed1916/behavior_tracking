import React, { useState, useRef, useEffect } from 'react';
import SegmentDisplay from './SegmentDisplay';

export default function LiveView({ model: propModel = '', prompt: propPrompt = '', classifierSource: propClassifierSource = 'llm', selectedSubtask = '', subtaskId = '', compareTimings = false, jpegQuality = 80, maxWidth = '', saveRecording = false, enableMediapipe = false, enableYolo = false, classifier: propClassifier = null, classifierMode: propClassifierMode = null, classifierPrompt: propClassifierPrompt = null }) {
  const [running, setRunning] = useState(false);
  const imgRef = useRef(null);
  const eventSourceRef = useRef(null);

  // UI options (model/useLLM come from props)
  const [model, setModel] = useState(propModel || '');
  const [classifierSource, setClassifierSource] = useState(propClassifierSource || 'llm');
  const [savedVideoUrl, setSavedVideoUrl] = useState('');
  const [classifiers, setClassifiers] = useState({});

  // Live metadata
  const [captionText, setCaptionText] = useState('');
  const [labelText, setLabelText] = useState('');
  const [llmOutput, setLlmOutput] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [segments, setSegments] = useState([]);

  useEffect(() => {
    // Keep state in sync if parent props change
    setModel(propModel || '');
    setClassifierSource(propClassifierSource || 'llm');
    // LiveView is now fully controlled: parent passes `classifierMode` and `classifierPrompt`.
      // fetch available label modes
    (async () => {
      try {
          const resp = await fetch('http://localhost:8001/backend/rules');
        if (!resp.ok) return;
        const j = await resp.json();
        const labelModes = j.label_modes || {};
        setClassifiers(labelModes);
      } catch (e) {
          console.warn('Failed to fetch label modes', e);
      }
    })();
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [propModel, propPrompt, propClassifierSource, selectedSubtask]);

  const buildUrl = () => {
    const base = `${window.location.protocol}//${window.location.hostname}:8001`;
    const params = new URLSearchParams();
    if (model) params.set('model', model);
    params.set('classifier_source', classifierSource || 'llm');
    // Use the parent-provided prompt (no local duplicate)
    if (propPrompt) params.set('prompt', propPrompt);
    if (subtaskId) params.set('subtask_id', subtaskId);
    if (compareTimings) params.set('compare_timings', 'true');
    if (jpegQuality) params.set('jpeg_quality', String(jpegQuality));
    if (maxWidth) params.set('max_width', String(maxWidth));
    if (saveRecording) params.set('save_video', 'true');
    if (enableMediapipe) params.set('enable_mediapipe', 'true');
    if (enableYolo) params.set('enable_yolo', 'true');
    // Use parent-controlled classifier mode/prompt (fallback to sensible defaults)
    const cm = propClassifierMode || 'binary';
    const cp = propClassifierPrompt || '';
    if (cm) params.set('classifier_mode', cm);
    if (cp) params.set('classifier_prompt', cp);
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
    setSavedVideoUrl('');
    setSegments([]);

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
        if (data.stage === 'segment') {
          // Append temporal segment to state
          setSegments(prev => [...prev, data]);
        }
        if (data.caption) setCaptionText(data.caption);
        if (data.label) setLabelText(data.label);
        if (data.llm_output) setLlmOutput(data.llm_output);
        const b64 = data.image || data.frame || data.jpeg || data.jpg;
        if (b64 && imgRef.current) {
          imgRef.current.src = `data:image/jpeg;base64,${b64}`;
        }
        if (data.stage === 'finished' && data.video_url) {
          // save the returned video URL for display
          setSavedVideoUrl(data.video_url);
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
          <div style={{ color: 'var(--muted)' }}><strong>Classifier:</strong> {classifierSource || 'llm'}</div>
          <div style={{ marginLeft: 12 }}>
            <div style={{ color: 'var(--muted)' }}><strong>Mode:</strong> {propClassifierMode || 'binary'}</div>
          </div>
        </div>

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
            <div style={{ marginBottom: 8 }}>
              <label style={{ display: 'block' }}>Label Prompt (optional):</label>
              <textarea value={propClassifierPrompt || ''} readOnly rows={3} style={{ width: '100%' }} placeholder="Override prompt template (controlled by parent)" />
            </div>
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
        {savedVideoUrl ? (
          <div style={{ marginTop: 10 }}>
            <strong>Saved Recording</strong>
            <div style={{ marginTop: 6 }}><a target="_blank" rel="noreferrer" href={savedVideoUrl}>{savedVideoUrl}</a></div>
          </div>
        ) : null}

        {segments.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <h5>Temporal Segments (LLM)</h5>
            <SegmentDisplay segments={segments} />
          </div>
        )}
      </div>
    </div>
  );
}
