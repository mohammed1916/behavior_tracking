import React, { useState, useRef, useEffect } from 'react';

export default function LiveView({ model: propModel = '', prompt: propPrompt = '', useLLM: propUseLLM = false, selectedSubtask = '', subtaskId = '', compareTimings = false, jpegQuality = 80, maxWidth = '', saveRecording = false, enableMediapipe = false, enableYolo = false, ruleSet: propRuleSet = null, classifier: propClassifier = null, classifierMode: propClassifierMode = null, classifierPrompt: propClassifierPrompt = null }) {
  const [running, setRunning] = useState(false);
  const imgRef = useRef(null);
  const eventSourceRef = useRef(null);

  // UI options (model/useLLM come from props)
  const [model, setModel] = useState(propModel || '');
  const [useLLM, setUseLLM] = useState(!!propUseLLM);
  const [savedVideoUrl, setSavedVideoUrl] = useState('');
  const [ruleSets, setRuleSets] = useState({});
  const [classifiers, setClassifiers] = useState({});
  const [ruleSet, setRuleSet] = useState(propRuleSet || 'default');
  const [classifier, setClassifier] = useState(propClassifier || 'blip_binary');
  const [classifierMode, setClassifierMode] = useState(propClassifierMode || 'binary');
  const [classifierPrompt, setClassifierPrompt] = useState(propClassifierPrompt || '');

  // Live metadata
  const [captionText, setCaptionText] = useState('');
  const [labelText, setLabelText] = useState('');
  const [llmOutput, setLlmOutput] = useState('');
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    // Keep state in sync if parent props change
    setModel(propModel || '');
    setUseLLM(!!propUseLLM);
    // if parent passed overrides, prefer them
    if (propRuleSet) setRuleSet(propRuleSet);
    if (propClassifier) setClassifier(propClassifier);
    if (propClassifierMode) setClassifierMode(propClassifierMode);
    if (propClassifierPrompt) setClassifierPrompt(propClassifierPrompt);
    // fetch rule sets & classifiers
    (async () => {
      try {
        const resp = await fetch(`${window.location.protocol}//${window.location.hostname}:8001/backend/rules`);
        if (!resp.ok) return;
        const j = await resp.json();
        setRuleSets(j.rule_sets || {});
        setClassifiers(j.classifiers || {});
        if (Object.keys(j.rule_sets || {}).length && !propRuleSet) setRuleSet(Object.keys(j.rule_sets)[0]);
        if (Object.keys(j.classifiers || {}).length && !propClassifier) setClassifier(Object.keys(j.classifiers)[0]);
      } catch (e) {
        console.warn('Failed to fetch rules', e);
      }
    })();
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
    if (saveRecording) params.set('save_video', 'true');
    if (enableMediapipe) params.set('enable_mediapipe', 'true');
    if (enableYolo) params.set('enable_yolo', 'true');
    // prefer props from parent when provided
    const rs = propRuleSet || ruleSet;
    const cl = propClassifier || classifier;
    const cm = propClassifierMode || classifierMode;
    const cp = propClassifierPrompt || classifierPrompt;
    if (rs) params.set('rule_set', rs);
    if (cl) params.set('classifier', cl);
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
          <div style={{ color: 'var(--muted)' }}><strong>Use LLM:</strong> {useLLM ? 'Yes' : 'No'}</div>
          <div style={{ marginLeft: 12 }}>
            <label style={{ color: 'var(--muted)' }}>Rule Set:
              <select value={ruleSet} onChange={(e) => setRuleSet(e.target.value)} style={{ marginLeft: 6 }}>
                {Object.keys(ruleSets).length > 0 ? Object.keys(ruleSets).map(k => <option key={k} value={k}>{k}</option>) : <option value="default">default</option>}
              </select>
            </label>
          </div>
          <div>
            <label style={{ color: 'var(--muted)' }}>Classifier:
              <select value={classifier} onChange={(e) => setClassifier(e.target.value)} style={{ marginLeft: 6 }}>
                {Object.keys(classifiers).length > 0 ? Object.keys(classifiers).map(k => <option key={k} value={k}>{k}</option>) : (
                  <>
                    <option value="blip_binary">blip_binary</option>
                    <option value="qwen_activity">qwen_activity</option>
                  </>
                )}
              </select>
            </label>
          </div>
          <div>
            <label style={{ color: 'var(--muted)' }}>Mode:
              <select value={classifierMode} onChange={(e) => setClassifierMode(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="binary">binary</option>
                <option value="multi">multi</option>
              </select>
            </label>
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
              <label style={{ display: 'block' }}>Classifier Prompt (optional):</label>
              <textarea value={classifierPrompt} onChange={(e) => setClassifierPrompt(e.target.value)} rows={3} style={{ width: '100%' }} placeholder="Override prompt template" />
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
      </div>
    </div>
  );
}
