import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showPose, setShowPose] = useState(false);
  const [mode, setMode] = useState('upload');

  // VLM state
  const [vlmModel, setVlmModel] = useState('gpt-4o-vlm');
  const [vlmAvailableModels, setVlmAvailableModels] = useState([]);
  const [vlmPrompt, setVlmPrompt] = useState('');
  const [vlmVideo, setVlmVideo] = useState(null);
  const [vlmResult, setVlmResult] = useState(null);
  const [vlmLoading, setVlmLoading] = useState(false);
  const [vlmUseLocal, setVlmUseLocal] = useState(true);
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

  const handleVlmSubmit = async () => {
    if (!vlmPrompt && !vlmVideo) {
      alert('Enter a prompt or select a video for the VLM.');
      return;
    }

    setVlmLoading(true);
    setVlmResult(null);

    try {
      const formData = new FormData();
      formData.append('model', vlmModel);
      formData.append('prompt', vlmPrompt);
      if (vlmVideo) formData.append('video', vlmVideo);

      const endpoint = vlmUseLocal ? 'http://localhost:8001/backend/vlm_local' : 'http://localhost:8001/backend/vlm';
      const resp = await fetch(endpoint, { method: 'POST', body: formData });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || resp.statusText);
      }
      const data = await resp.json();
      setVlmResult(data);
    } catch (err) {
      console.error('VLM request failed', err);
      setVlmResult({ error: err.message });
    } finally {
      setVlmLoading(false);
    }
  };

  const handleLlmCheck = async () => {
    if (!llmText || llmText.trim().length === 0) return alert('Enter text to check');
    setLlmLoading(true);
    setLlmResult(null);
    try {
      const form = new FormData();
      form.append('text', llmText);
      form.append('max_context', String(llmMaxContext));
      const resp = await fetch('http://localhost:8001/backend/llm_length_check', { method: 'POST', body: form });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      setLlmResult(data);
    } catch (e) {
      setLlmResult({ error: e.message });
    } finally {
      setLlmLoading(false);
    }
  };

  // Fetch local models when user switches to local VLM
  useEffect(() => {
    let abort = false;
    async function fetchLocalModels() {
      if (!vlmUseLocal) return setVlmAvailableModels([]);
      try {
        const resp = await fetch('http://localhost:8001/backend/vlm_local_models');
        if (!resp.ok) return setVlmAvailableModels([]);
        const data = await resp.json();
        if (abort) return;
        setVlmAvailableModels(data.models || []);
        if ((data.models || []).length > 0) setVlmModel(data.models[0].id);
      } catch (e) {
        console.warn('Could not fetch local VLM models', e);
        setVlmAvailableModels([]);
      }
    }
    fetchLocalModels();
    return () => { abort = true; };
  }, [vlmUseLocal]);

  const handleUpload = async () => {
    if (!file) return alert('Please select a file first!');
    setLoading(true);
    setStatus('Uploading and processing...');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch('http://localhost:8001/backend/analyze_video', { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Error: ${response.statusText}`);
      const data = await response.json();
      setResult(data);
      setStatus('Processing complete!');
    } catch (error) {
      console.error('Error uploading file:', error);
      setStatus(`Failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Behavior Tracking Analysis</h1>

      <div className="controls">
        <div className="panel left">
          <div style={{ marginBottom: 12 }} className="mode-toggle">
            <input id="mode-upload" type="radio" name="mode" value="upload" checked={mode === 'upload'} onChange={() => { setMode('upload'); setShowPose(false); }} />
            <label htmlFor="mode-upload">Upload</label>
            <input id="mode-stream" type="radio" name="mode" value="stream" checked={mode === 'stream'} onChange={() => { setMode('stream'); setShowPose(false); }} />
            <label htmlFor="mode-stream">Live</label>
          </div>

          {mode === 'upload' && (
            <div className="upload-section">
              <input type="file" accept="video/*" onChange={handleFileChange} />
              <button onClick={handleUpload} disabled={loading || !file}>{loading ? 'Processing...' : 'Analyze Video'}</button>
            </div>
          )}

          {mode === 'stream' && (
            <div className="stream-section">
              <div style={{ marginBottom: 8 }}>
                <button onClick={() => setShowPose((s) => !s)}>{showPose ? 'Stop Stream' : 'Start Stream'}</button>
              </div>
              {showPose && (
                <div className="pose-stream">
                  <h3>Live Pose</h3>
                  <img src="http://localhost:8001/backend/stream_pose" alt="Live Pose" style={{ maxWidth: '100%' }} onError={() => setShowPose(false)} />
                </div>
              )}
            </div>
          )}

          <div style={{ marginTop: 14 }} className="panel">
            <h3>Results</h3>
            {status && <p className="status">{status}</p>}
            {result ? (
              <>
                <p><strong>Task Completed:</strong> {result.task_completed ? 'YES' : 'NO'}</p>
                {result.download_url && (
                  <div className="video-container">
                    <h4>Processed Video</h4>
                    <video controls width="100%"><source src={`http://localhost:8001${result.download_url}`} type="video/mp4" /></video>
                    <div style={{ marginTop: 8 }}><a href={`http://localhost:8001${result.download_url}`} download><button>Download Processed Video</button></a></div>
                  </div>
                )}
              </>
            ) : (
              <p>No results yet.</p>
            )}
          </div>
        </div>

        <div className="panel right">
          <h3>VLM (Video)</h3>
          <div className="vlm-section">
            <label>Model:
              <select value={vlmModel} onChange={(e) => setVlmModel(e.target.value)}>
                {vlmUseLocal ? (
                  vlmAvailableModels.length > 0 ? (
                    vlmAvailableModels.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)
                  ) : (
                    <option value="">(no local models available)</option>
                  )
                ) : (
                  <>
                    <option value="gpt-4o-vlm">gpt-4o-vlm (remote placeholder)</option>
                    <option value="gpt-5-vlm">gpt-5-vlm (remote placeholder)</option>
                    <option value="openai-vision">openai-vision (remote placeholder)</option>
                  </>
                )}
              </select>
            </label>

            <label>Prompt:
              <textarea value={vlmPrompt} onChange={(e) => setVlmPrompt(e.target.value)} placeholder="Ask about the video or request an analysis" rows={3} />
            </label>

            <label>Optional Video:
              <input type="file" accept="video/*" onChange={handleVlmVideoChange} />
            </label>

            <label style={{ display: 'block', marginTop: 8 }}>
              <input type="checkbox" checked={vlmUseLocal} onChange={(e) => setVlmUseLocal(e.target.checked)} /> Use local VLM
            </label>
            <div style={{ marginTop: 6 }}>
              {vlmUseLocal ? (
                <small style={{ color: '#666' }}>{vlmAvailableModels.length > 0 ? `Using local model: ${vlmAvailableModels[0].name}` : 'No local VLM models detected on the server.'}</small>
              ) : (
                <small style={{ color: '#666' }}>Remote model identifiers (placeholders for cloud VLMs).</small>
              )}
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
                                            <button onClick={() => playRange(r.startTime, r.endTime)}>Jump</button>
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
                                          <button onClick={() => playRange(r.startTime, r.endTime)}>Jump</button>
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
                  </>
                )}
              </div>
            )}
            {/* LLM Length Check */}
            <div style={{ marginTop: 16, textAlign: 'left' }}>
              <h4>LLM Length Check</h4>
              <label style={{ display: 'block', marginBottom: 6 }}>
                Text to check:
                <textarea value={llmText} onChange={(e) => setLlmText(e.target.value)} rows={4} style={{ width: '100%', marginTop: 6 }} />
              </label>
              <label style={{ display: 'block', marginBottom: 8 }}>
                Max context tokens:
                <input type="number" value={llmMaxContext} onChange={(e) => setLlmMaxContext(Number(e.target.value))} style={{ width: 120, marginLeft: 8 }} />
              </label>
              <div>
                <button onClick={handleLlmCheck} disabled={llmLoading}>{llmLoading ? 'Checking...' : 'Check Length'}</button>
              </div>
              {llmResult && (
                <div style={{ marginTop: 8 }}>
                  <strong>Result:</strong>
                  <pre style={{ whiteSpace: 'pre-wrap', textAlign: 'left', background: '#f6f7fb', padding: 8, borderRadius: 6 }}>{JSON.stringify(llmResult, null, 2)}</pre>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

