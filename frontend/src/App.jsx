import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import StoredAnalyses from './components/StoredAnalyses';
import AnalysisDetails from './components/AnalysisDetails';
import FileUpload from './components/FileUpload';
import LiveView from './components/LiveView';
import Tasks from './components/Tasks';
import Subtasks from './components/Subtasks';

function App() {
  // const [file, setFile] = useState(null);
  // const [status, setStatus] = useState('');
  // const [result, setResult] = useState(null);
  // const [loading, setLoading] = useState(false);
  // const [showPose, setShowPose] = useState(false);
  // const [mode, setMode] = useState('upload');

  // VLM state
  const [vlmModel, setVlmModel] = useState('qwen_local');
  const [vlmAvailableModels, setVlmAvailableModels] = useState([]);
  const [preloadedDevices, setPreloadedDevices] = useState({});
  const [modelLoading, setModelLoading] = useState(false);
  const [vlmLoadDevice, setVlmLoadDevice] = useState('auto');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advSubtaskId, setAdvSubtaskId] = useState('');
  const [advCompareTimings, setAdvCompareTimings] = useState(false);
  const [advJpegQuality, setAdvJpegQuality] = useState(80);
  const [advMaxWidth, setAdvMaxWidth] = useState('');
  const [advSaveRecording, setAdvSaveRecording] = useState(false);
  const [vlmPrompt, setVlmPrompt] = useState('');
  const [vlmVideo, setVlmVideo] = useState(null);
  const [vlmResult, setVlmResult] = useState(null);
  const [vlmLoading, setVlmLoading] = useState(false);
  const [vlmStream, setVlmStream] = useState(null);
  const [vlmUseLLM, setVlmUseLLM] = useState(false);
  const [ruleSets, setRuleSets] = useState({});
  const [classifiers, setClassifiers] = useState({});
  const [vlmRuleSet, setVlmRuleSet] = useState('default');
  const [vlmClassifier, setVlmClassifier] = useState('blip_binary');
  const [vlmClassifierMode, setVlmClassifierMode] = useState('binary');
  const [vlmClassifierPrompt, setVlmClassifierPrompt] = useState('');
  const [viewAnalysisId, setViewAnalysisId] = useState(null);
  const [enableMediapipe, setEnableMediapipe] = useState(false);
  const [enableYolo, setEnableYolo] = useState(false);
  // LLM length check state
  const [llmText, setLlmText] = useState('');
  const [llmMaxContext, setLlmMaxContext] = useState(2048);
  const [llmResult, setLlmResult] = useState(null);
  const [llmLoading, setLlmLoading] = useState(false);
  const vlmVideoRef = useRef(null);
  const pauseTimerRef = useRef(null);

  function computeRanges(frames, samples, fps) {
    if (!frames || frames.length === 0) return [];
    const sampleMap = {};
    (samples || []).forEach(s => { if (s && s.frame_index !== undefined) sampleMap[s.frame_index] = s.time_sec; });
    const times = frames.map(f => ({ frame: f, time: sampleMap[f] !== undefined ? sampleMap[f] : (fps ? (f / fps) : 0) }));
    times.sort((a, b) => a.time - b.time);
    const dtEst = (() => {
      if ((samples || []).length >= 2) {
        const s0 = samples[0].time_sec || 0;
        const s1 = samples[1].time_sec || 0;
        const d = Math.abs(s1 - s0);
        return d > 0 ? d : 1.0;
      }
      return 1.0;
    })();
    const maxGap = Math.max(1.0, dtEst * 1.5);
    const ranges = [];
    let cur = { startFrame: times[0].frame, endFrame: times[0].frame, startTime: times[0].time, endTime: times[0].time };
    for (let i = 1; i < times.length; i++) {
      const t = times[i];
      if ((t.time - cur.endTime) <= maxGap) {
        cur.endFrame = t.frame;
        cur.endTime = t.time;
      } else {
        ranges.push({ ...cur });
        cur = { startFrame: t.frame, endFrame: t.frame, startTime: t.time, endTime: t.time };
      }
    }
    ranges.push({ ...cur });
    return ranges;
  }

  function playRange(startSec, endSec) {
    const v = vlmVideoRef.current;
    if (!v) return;
    if (pauseTimerRef.current) { clearTimeout(pauseTimerRef.current); pauseTimerRef.current = null; }
    v.currentTime = startSec || 0;
    const dur = Math.max(0.2, (endSec || (startSec + 1)) - startSec);
    v.play().catch(() => {});
    pauseTimerRef.current = setTimeout(() => { try { v.pause(); } catch (e) {} pauseTimerRef.current = null; }, Math.ceil(dur * 1000) + 150);
  }

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setStatus('');
    setResult(null);
  };

  const handleVlmVideoChange = (e) => {
    setVlmVideo(e.target.files[0] || null);
    setVlmResult(null);
  };

  const [activeView, setActiveView] = useState('vlm');
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedSubtasks, setSelectedSubtasks] = useState([]);
  const [subtasksRefreshTrigger, setSubtasksRefreshTrigger] = useState(0);

  const onSubtaskSelect = useCallback((subtasks) => {
    setSelectedSubtasks(subtasks);
  }, []);

  const handleVlmSubmit = async () => {
    if (!vlmPrompt && !vlmVideo) {
      alert('Enter a prompt or select a video for the VLM.');
      return;
    }

    setVlmLoading(true);
    setVlmResult(null);

    try {
      // upload first
      const fd = new FormData();
      if (vlmVideo) fd.append('video', vlmVideo);
      const up = await fetch('http://localhost:8001/backend/upload_vlm', { method: 'POST', body: fd });
      if (!up.ok) throw new Error('Upload failed');
      const upj = await up.json();
      const filename = upj.filename;

      const url = `http://localhost:8001/backend/vlm_local_stream?filename=${encodeURIComponent(filename)}&model=${encodeURIComponent(vlmModel)}&prompt=${encodeURIComponent(vlmPrompt)}&use_llm=${vlmUseLLM ? 'true' : 'false'}${enableMediapipe ? '&enable_mediapipe=true' : ''}${enableYolo ? '&enable_yolo=true' : ''}${selectedSubtasks.length > 0 ? `&subtask_id=${encodeURIComponent(selectedSubtasks[0].id)}&compare_timings=true` : ''}`;
      if (vlmStream) { try { vlmStream.close(); } catch {} }
      const es = new EventSource(url);
      setVlmStream(es);

      const analysis = { model: vlmModel, prompt: vlmPrompt, samples: [], idle_frames: [], work_frames: [], fps: 30 };
      setVlmResult({ message: 'streaming', analysis });

      es.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (!data) return;
          if (data.stage === 'video_info') {
            analysis.fps = data.fps || analysis.fps;
            analysis.video_info = data;
            setVlmResult({ message: 'streaming', analysis: { ...analysis } });
          } else if (data.stage === 'sample') {
            const sample = { frame_index: data.frame_index, time_sec: data.time_sec, caption: data.caption, label: data.label };
            analysis.samples.push(sample);
            if (data.llm_output) {
              analysis.samples[analysis.samples.length - 1].llm_output = data.llm_output;
            }
            if (data.label === 'idle') analysis.idle_frames.push(data.frame_index);
            if (data.label === 'work') analysis.work_frames.push(data.frame_index);
            setVlmResult({ message: 'streaming', analysis: { ...analysis } });
          } else if (data.stage === 'sample_error') {
            analysis.samples.push({ frame_index: data.frame_index, error: data.error });
            setVlmResult({ message: 'streaming', analysis: { ...analysis } });
          } else if (data.stage === 'finished') {
            analysis.video_url = data.video_url;
            setVlmResult({ message: 'done', analysis: { ...analysis } });
            setVlmLoading(false);
            try { es.close(); } catch {}
            setVlmStream(null);
            // If server saved analysis and returned stored id, surface link
            if (data.stored_analysis_id) {
              setViewAnalysisId(data.stored_analysis_id);
              setActiveView('stored');
            }
            // Refresh subtasks in case counts were updated
            setSubtasksRefreshTrigger(prev => prev + 1);
          } else if (data.stage === 'error') {
            analysis.error = data.message;
            setVlmResult({ message: 'error', analysis: { ...analysis } });
            setVlmLoading(false);
            try { es.close(); } catch {}
            setVlmStream(null);
          }
        } catch (e) {
          console.error('parse sse', e);
        }
      };

      es.onerror = (err) => {
        console.warn('SSE error', err);
        setVlmLoading(false);
        try { es.close(); } catch {}
        setVlmStream(null);
      };
    } catch (err) {
      console.error('VLM request failed', err);
      setVlmResult({ error: err.message || String(err) });
      setVlmLoading(false);
    }
  };

  // Fetch local models
  const fetchLocalModels = async () => {
    try {
      const resp = await fetch('http://localhost:8001/backend/vlm_local_models');
      if (!resp.ok) {
        setVlmAvailableModels([]);
        return;
      }
      const data = await resp.json();
      const models = data.models || [];
      setVlmAvailableModels(models);
      if (models.length > 0) {
        const exists = models.some(m => m.id === vlmModel);
        if (!vlmModel || !exists) {
          setVlmModel(models[0].id);
        }
      }
    } catch (e) {
      console.warn('Could not fetch local VLM models', e);
      setVlmAvailableModels([]);
    }
  };

  useEffect(() => {
    fetchLocalModels();
    fetchPreloadedModels();
    // fetch available rule sets and classifiers
    (async () => {
      try {
        const resp = await fetch('http://localhost:8001/backend/rules');
        if (!resp.ok) return;
        const j = await resp.json();
        setRuleSets(j.rule_sets || {});
        setClassifiers(j.classifiers || {});
        // keep defaults if current selections aren't present
        if (!vlmRuleSet && Object.keys(j.rule_sets || {}).length) setVlmRuleSet(Object.keys(j.rule_sets)[0]);
        if (!vlmClassifier && Object.keys(j.classifiers || {}).length) setVlmClassifier(Object.keys(j.classifiers)[0]);
      } catch (e) {
        console.warn('Failed to fetch rule sets', e);
      }
    })();
  }, []);

  const fetchPreloadedModels = async () => {
    try {
      const resp = await fetch('http://localhost:8001/backend/preloaded_models');
      if (!resp.ok) return setPreloadedDevices({});
      const data = await resp.json();
      const models = data.models || {};
      // models is a map id -> {type, loaded, device}
      setPreloadedDevices(models);
    } catch (e) {
      console.warn('Could not fetch preloaded models', e);
      setPreloadedDevices({});
    }
  };

  const loadModel = async () => {
    if (!vlmModel) return alert('Select a model first');
    setModelLoading(true);
    try {
      const form = new FormData();
      form.append('model', vlmModel);
      if (vlmLoadDevice && vlmLoadDevice !== 'auto') form.append('device', vlmLoadDevice);
      const resp = await fetch('http://localhost:8001/backend/load_vlm_model', { method: 'POST', body: form });
      const data = await resp.json();
      if (!resp.ok || !data.loaded) {
        alert('Model load failed: ' + (data.alert || data.error || resp.statusText));
      } else {
        alert('Model loaded: ' + data.model + (data.device ? ' (device: ' + data.device + ')' : ''));
        // refresh list to reflect loaded status (if server updates it)
        fetchLocalModels();
        fetchPreloadedModels();
      }
    } catch (e) {
      alert('Load model error: ' + (e.message || String(e)));
    } finally {
      setModelLoading(false);
    }
  };

  // derive selected local model's display name
  const selectedVlmModelName = (vlmAvailableModels || []).find(m => m.id === vlmModel)?.name || (vlmAvailableModels && vlmAvailableModels[0] && vlmAvailableModels[0].name) || '';

  useEffect(() => {
    console.log("selectedVlmModelName changed:", selectedVlmModelName);
  }, [selectedVlmModelName]);
  // const handleUpload = async () => {
  //   if (!file) return alert('Please select a file first!');
  //   setLoading(true);
  //   setStatus('Uploading and processing...');
  //   const formData = new FormData();
  //   formData.append('file', file);
  //   try {
  //     const response = await fetch('http://localhost:8001/backend/analyze_video', { method: 'POST', body: formData });
  //     if (!response.ok) throw new Error(`Error: ${response.statusText}`);
  //     const data = await response.json();
  //     setResult(data);
  //     setStatus('Processing complete!');
  //   } catch (error) {
  //     console.error('Error uploading file:', error);
  //     setStatus(`Failed: ${error.message}`);
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  return (
    <div className="App">
      <h1>Behavior Tracking Analysis</h1>

      <nav className="navbar">
        <button className={activeView === 'vlm' ? 'active' : ''} onClick={() => setActiveView('vlm')}>VLM</button>
        <button className={activeView === 'stored' ? 'active' : ''} onClick={() => setActiveView('stored')}>Stored Analyses</button>
        <button className={activeView === 'tasks' ? 'active' : ''} onClick={() => setActiveView('tasks')}>Tasks</button>
        <button className={activeView === 'subtasks' ? 'active' : ''} onClick={() => setActiveView('subtasks')}>Subtasks</button>
      </nav>

      <div className="content">
        {activeView === 'vlm' && (
          <div className="vlm-section">
            <h3>VLM (Video)</h3>
            <label>Model:
              <select value={vlmModel} onChange={(e) => setVlmModel(e.target.value)}>
                {vlmAvailableModels.length > 0 ? (
                  vlmAvailableModels.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)
                ) : (
                  <option value="">(no local models available)</option>
                )}
              </select>
              <select value={vlmLoadDevice} onChange={(e) => setVlmLoadDevice(e.target.value)} style={{ marginLeft: 8 }} title="Device to load model on">
                <option value="auto">Auto</option>
                <option value="cpu">CPU</option>
                <option value="cuda:0">GPU (cuda:0)</option>
              </select>
              <div style={{  margin: 30 }}/>
              <button type="button" onClick={loadModel} disabled={!vlmModel || modelLoading} style={{ marginLeft: 8 }}>{modelLoading ? 'Loading...' : 'Load model'}</button>
              <button type="button" onClick={() => { fetchLocalModels(); fetchPreloadedModels(); }} style={{ marginLeft: 8 }}>Refresh models</button>
              
            </label>

              <div style={{ margin: 16 }}>
                <button onClick={() => setShowAdvanced(s => !s)}>{showAdvanced ? 'Hide Advanced' : 'Show Advanced'}</button>
                {showAdvanced && (
                  <div style={{ marginTop: 8, padding: 8, border: '1px dashed var(--panel-border)', borderRadius: 6 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div>
                        <label>Subtask ID (optional):</label>
                        <input style={{ width: '100%' }} value={advSubtaskId} onChange={(e) => setAdvSubtaskId(e.target.value)} placeholder="subtask uuid" />
                      </div>
                      <div>
                        <label>Compare Timings:</label>
                        <input type="checkbox" checked={advCompareTimings} onChange={(e) => setAdvCompareTimings(e.target.checked)} />
                      </div>
                      <div>
                        <label>JPEG Quality: {advJpegQuality}</label>
                        <input type="range" min={10} max={95} value={advJpegQuality} onChange={(e) => setAdvJpegQuality(parseInt(e.target.value || '80'))} />
                      </div>
                      <div>
                        <label>Max Width (px):</label>
                        <input style={{ width: '100%' }} value={advMaxWidth} onChange={(e) => setAdvMaxWidth(e.target.value)} placeholder="e.g. 640" />
                      </div>
                      <div>
                        <label>Save Recording:</label>
                        <input type="checkbox" checked={advSaveRecording} onChange={(e) => setAdvSaveRecording(e.target.checked)} />
                      </div>
                      <div>
                        <label>Enable MediaPipe:</label>
                        <input type="checkbox" checked={enableMediapipe} onChange={(e) => setEnableMediapipe(e.target.checked)} />
                      </div>
                      <div>
                        <label>Enable YOLO:</label>
                        <input type="checkbox" checked={enableYolo} onChange={(e) => setEnableYolo(e.target.checked)} />
                      </div>
                    </div>
                  </div>
                )}
              </div>

            <label>Prompt:
              <textarea value={vlmPrompt} onChange={(e) => setVlmPrompt(e.target.value)} placeholder="Ask about the video or request an analysis" rows={3} />
            </label>
              <div style={{ marginTop: 8 }}>
                <label style={{ display: 'block' }}>Rule Set:
                  <select value={vlmRuleSet} onChange={(e) => setVlmRuleSet(e.target.value)}>
                    {Object.keys(ruleSets).length > 0 ? Object.keys(ruleSets).map(k => <option key={k} value={k}>{k}</option>) : <option value="default">default</option>}
                  </select>
                </label>

                <label style={{ display: 'block', marginTop: 6 }}>Classifier:
                  <select value={vlmClassifier} onChange={(e) => setVlmClassifier(e.target.value)}>
                    {Object.keys(classifiers).length > 0 ? Object.keys(classifiers).map(k => <option key={k} value={k}>{k}</option>) : (
                      <>
                        <option value="blip_binary">blip_binary</option>
                        <option value="qwen_activity">qwen_activity</option>
                      </>
                    )}
                  </select>
                </label>

                <label style={{ display: 'block', marginTop: 6 }}>Classifier Mode:
                  <select value={vlmClassifierMode} onChange={(e) => setVlmClassifierMode(e.target.value)}>
                    <option value="binary">binary (work/idle)</option>
                    <option value="multi">multi (preserve adapter labels)</option>
                  </select>
                </label>

                <label style={{ display: 'block', marginTop: 6 }}>Classifier Prompt (optional override):
                  <textarea value={vlmClassifierPrompt} onChange={(e) => setVlmClassifierPrompt(e.target.value)} placeholder="Optional: override prompt template" rows={3} />
                </label>
              </div>

            <label style={{ display: 'block', marginTop: 8 }}>
              <input type="checkbox" checked={vlmUseLLM} onChange={(e) => setVlmUseLLM(e.target.checked)} /> Use LLM classifier for labels
              <small style={{ color: '#666', marginLeft: 8 }}>When enabled, a local text LLM (if available) will be used to decide work vs idle.</small>
            </label>
            <br/>

            <div style={{ marginTop: 8, border: '1px dotted var(--accent)', padding: '8px', paddingTop: 16 }}>
              <div className="tabs" style={{ display: 'flex', gap: 8, justifyContent: 'center'}}>
                <button className={`tab ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>Upload</button>
                <button className={`tab ${activeTab === 'live' ? 'active' : ''}`} onClick={() => setActiveTab('live')}>Live</button>
              </div>

              <div className="dotted-hr"/>

              <div style={{ marginTop: 10, justifyContent: 'center', display: 'flex' }}>
                {activeTab === 'upload' ? (
                  <FileUpload accept="video/*" onFileSelect={(f) => setVlmVideo(f)} initialFile={vlmVideo} label="Select a video for analysis" />
                ) : (
                  <LiveView
                    model={vlmModel}
                    prompt={vlmPrompt}
                    useLLM={vlmUseLLM}
                    ruleSet={vlmRuleSet}
                    classifier={vlmClassifier}
                    classifierMode={vlmClassifierMode}
                    classifierPrompt={vlmClassifierPrompt}
                    selectedSubtask={selectedSubtasks.length > 0 ? selectedSubtasks[0].id : ''}
                    subtaskId={advSubtaskId}
                    compareTimings={advCompareTimings}
                    jpegQuality={advJpegQuality}
                    maxWidth={advMaxWidth}
                    saveRecording={advSaveRecording}
                    enableMediapipe={enableMediapipe}
                    enableYolo={enableYolo}
                  />
                )}
              </div>
            </div>

            {selectedSubtasks.length > 0 && (
              <div style={{ marginTop: 10, padding: 10, backgroundColor: '#f0f9ff', borderRadius: 4 }}>
                <strong>Selected Subtasks:</strong> {selectedSubtasks.map(t => `${t.id} (${t.completed ? 'Completed' : 'Pending'})`).join(', ')}
                <button onClick={() => setSelectedSubtasks([])} style={{ marginLeft: 8 }}>Clear</button>
              </div>
            )}


            <div style={{ marginTop: 6 }}>
              <small style={{ color: '#666' }}>
                {vlmAvailableModels.length > 0 ? (
                  (() => {
                    const meta = preloadedDevices && preloadedDevices[vlmModel];
                    const dev = meta && meta.device;
                    const devLabel = dev ? (String(dev).toLowerCase().includes('cuda') ? 'GPU' : 'CPU') : 'unknown';
                    const statusMsg = meta && meta.device_status_message;
                    return (
                      <>
                        {`Using local model: ${selectedVlmModelName} (device: ${devLabel})`}
                        {statusMsg ? <div style={{ color: '#a00', marginTop: 6, whiteSpace: 'pre-wrap' }}>{statusMsg}</div> : null}
                      </>
                    );
                  })()
                ) : 'No local VLM models detected on the server.'}
              </small>
            </div>

            <div style={{ marginTop: 8 }}>
              <button onClick={handleVlmSubmit} disabled={vlmLoading}>{vlmLoading ? 'Running...' : 'Run VLM on Video'}</button>
            </div>

            {vlmResult && (
              <div className="vlm-result" style={{ marginTop: 12 }}>
                <h4>VLM Result</h4>
                {vlmResult.error ? (
                  <pre style={{ color: 'red' }}>{vlmResult.error}</pre>
                ) : (
                  <>
                    <div style={{ marginBottom: 8 }}><strong>Analysis</strong><pre className="vlm-result">{JSON.stringify(vlmResult.analysis, null, 2)}</pre></div>
                    {vlmResult.analysis && (vlmResult.analysis.captions || vlmResult.analysis.caption) && (
                      <div><strong>Captions</strong><ul className="captions-list">{vlmResult.analysis.captions ? vlmResult.analysis.captions.map((c, i) => <li key={i}>{c}</li>) : <li>{vlmResult.analysis.caption}</li>}</ul></div>
                    )}
                    {vlmResult.analysis && vlmResult.analysis.video_url && (
                      <div style={{ marginTop: 8 }}>
                        <strong>Preview</strong>
                        <video ref={vlmVideoRef} controls width="100%" style={{ marginTop: 8 }}>
                          <source src={`http://localhost:8001${vlmResult.analysis.video_url}`} type="video/mp4" />
                        </video>

                        <div style={{ display: 'flex', gap: 12, marginTop: 10 }}>
                          <div style={{ flex: 1, textAlign: 'left' }}>
                            <strong>Idle Frames</strong>
                              {vlmResult.analysis.idle_frames && vlmResult.analysis.idle_frames.length > 0 ? (
                                <div className="captions-list">
                                  {(() => {
                                    const fps = vlmResult.analysis.fps || 30;
                                    const ranges = computeRanges(vlmResult.analysis.idle_frames, vlmResult.analysis.samples || [], fps);
                                    return ranges.slice(0,50).map((r, i) => {
                                      const captions = (vlmResult.analysis.samples || []).filter(s => s.time_sec >= r.startTime - 0.0001 && s.time_sec <= r.endTime + 0.0001).map(s => s.caption).filter(Boolean);
                                      return (
                                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                          <div style={{ flex: 1 }}>
                                            <small>{r.startTime.toFixed(2)}s - {r.endTime.toFixed(2)}s</small>
                                            <div style={{ fontSize: 12, color: '#444' }}>{captions.length ? captions.join(' | ') : ''}</div>
                                          </div>
                                          <div>
                                            <button onClick={() => playRange(r.startTime, r.endTime)}>Play</button>
                                          </div>
                                        </div>
                                      )
                                    })
                                  })()}
                                </div>
                              ) : (<div style={{ color: '#666' }}>No idle frames detected.</div>)}
                          </div>

                          <div style={{ flex: 1, textAlign: 'left' }}>
                            <strong>Work Frames</strong>
                            {vlmResult.analysis.work_frames && vlmResult.analysis.work_frames.length > 0 ? (
                              <div className="captions-list">
                                {(() => {
                                  const fps = vlmResult.analysis.fps || 30;
                                  const ranges = computeRanges(vlmResult.analysis.work_frames, vlmResult.analysis.samples || [], fps);
                                  return ranges.slice(0,50).map((r, i) => {
                                    const captions = (vlmResult.analysis.samples || []).filter(s => s.time_sec >= r.startTime - 0.0001 && s.time_sec <= r.endTime + 0.0001).map(s => s.caption).filter(Boolean);
                                    return (
                                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                        <div style={{ flex: 1 }}>
                                          <small>{r.startTime.toFixed(2)}s - {r.endTime.toFixed(2)}s</small>
                                          <div style={{ fontSize: 12, color: '#444' }}>{captions.length ? captions.join(' | ') : ''}</div>
                                        </div>
                                        <div>
                                          <button onClick={() => playRange(r.startTime, r.endTime)}>Play</button>
                                        </div>
                                      </div>
                                    )
                                  })
                                })()}
                              </div>
                            ) : (<div style={{ color: '#666' }}>No work frames detected.</div>)}
                          </div>
                        </div>
                      </div>
                    )}
                    {/* Stored analyses panel is rendered in the left controls now */}
                  </>
                )}
              </div>
            )}
          </div>
        )}

        {activeView === 'stored' && (
          <div style={{ display: 'flex', gap: 12 }}>
            <div style={{ flex: 1 }}>
              <StoredAnalyses onView={(id) => setViewAnalysisId(id)} />
            </div>
            <div style={{ width: 520 }}>
              <AnalysisDetails analysisId={viewAnalysisId} onClose={() => setViewAnalysisId(null)} />
            </div>
          </div>
        )}

        {activeView === 'tasks' && (
          <Tasks />
        )}

        {activeView === 'subtasks' && (
          <Subtasks onSubtaskSelect={onSubtaskSelect} refreshTrigger={subtasksRefreshTrigger} />
        )}
      </div>
    </div>
  );
}

export default App;

