import React, { useState, useEffect } from 'react';

export default function Training() {
    const [activeTab, setActiveTab] = useState('features');
    const [features, setFeatures] = useState([]);
    const [datasets, setDatasets] = useState([]);
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Selection states
    const [selectedFeatures, setSelectedFeatures] = useState([]);
    const [selectedDataset, setSelectedDataset] = useState(null);

    // Params
    const [buildParams, setBuildParams] = useState({
        window_sec: 2.0,
        step_sec: 0.5,
        balance_method: 'downsample',
        test_ratio: 0.15
    });

    const [trainParams, setTrainParams] = useState({
        model_type: 'rf',
        n_estimators: 100,
        aggregation: 'stats'
    });

    const fetchData = async () => {
        setLoading(true);
        try {
            const [fRes, dRes, mRes] = await Promise.all([
                fetch('http://localhost:8001/backend/features'),
                fetch('http://localhost:8001/backend/datasets'),
                fetch('http://localhost:8001/backend/models')
            ]);

            const fData = await fRes.json();
            const dData = await dRes.json();
            const mData = await mRes.json();

            setFeatures(fData.features || []);
            setDatasets(dData.datasets || []);
            setModels(mData.models || []);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    const handleBuildDataset = async () => {
        if (selectedFeatures.length === 0) return alert('Select at least one feature file');
        setLoading(true);
        try {
            const fd = new FormData();
            fd.append('feature_files', selectedFeatures.join(','));
            fd.append('window_sec', buildParams.window_sec);
            fd.append('step_sec', buildParams.step_sec);
            fd.append('balance_method', buildParams.balance_method);
            fd.append('test_ratio', buildParams.test_ratio);

            const res = await fetch('http://localhost:8001/backend/build_task_dataset', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok) throw new Error(data.alert || 'Build failed');

            alert(`Dataset built!\nTrain samples: ${data.train_size}\nTest samples: ${data.test_size}`);
            fetchData();
            setActiveTab('datasets');
        } catch (e) {
            alert(e.message);
        } finally {
            setLoading(false);
        }
    };

    const handleTrainModel = async () => {
        if (!selectedDataset) return alert('Select a dataset');
        setLoading(true);
        try {
            const fd = new FormData();
            fd.append('dataset_dir', selectedDataset);
            fd.append('model_type', trainParams.model_type);
            fd.append('n_estimators', trainParams.n_estimators);
            fd.append('aggregation', trainParams.aggregation);

            const res = await fetch('http://localhost:8001/backend/train_task_model', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok) throw new Error(data.alert || 'Training failed');

            alert(`Model trained!\n\nMetrics:\n${JSON.stringify(data.metrics, null, 2)}`);
            fetchData();
            setActiveTab('models');
        } catch (e) {
            alert(e.message);
        } finally {
            setLoading(false);
        }
    };

    const toggleFeature = (fname) => {
        setSelectedFeatures(prev =>
            prev.includes(fname) ? prev.filter(f => f !== fname) : [...prev, fname]
        );
    };

    return (
        <div className="training-view" style={{ padding: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 20 }}>
                <h2>Model Training</h2>
                <button onClick={fetchData} disabled={loading}>{loading ? 'Refreshing...' : 'Refresh Data'}</button>
            </div>

            <div className="tabs" style={{ display: 'flex', gap: 10, marginBottom: 20, borderBottom: '1px solid var(--panel-border)' }}>
                {['features', 'datasets', 'models'].map(t => (
                    <button
                        key={t}
                        onClick={() => setActiveTab(t)}
                        style={{
                            padding: '10px 20px',
                            border: 'none',
                            background: activeTab === t ? 'var(--accent)' : 'transparent',
                            color: activeTab === t ? 'white' : 'var(--text)',
                            cursor: 'pointer',
                            borderRadius: '4px 4px 0 0'
                        }}
                    >
                        {t.charAt(0).toUpperCase() + t.slice(1)}
                    </button>
                ))}
            </div>

            {error && <div style={{ color: 'red', marginBottom: 20 }}>Error: {error}</div>}

            {/* FEATURES TAB */}
            {activeTab === 'features' && (
                <div>
                    <div style={{ display: 'flex', gap: 20 }}>
                        <div style={{ flex: 2 }}>
                            <h4>Available Feature Files (extracted from videos)</h4>
                            <div style={{ maxHeight: '60vh', overflow: 'auto', border: '1px solid var(--panel-border)', borderRadius: 4 }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                    <thead>
                                        <tr style={{ background: 'var(--panel-bg)', textAlign: 'left' }}>
                                            <th style={{ padding: 8 }}><input type="checkbox" onChange={(e) => setSelectedFeatures(e.target.checked ? features.map(f => f.filename) : [])} checked={selectedFeatures.length === features.length && features.length > 0} /></th>
                                            <th style={{ padding: 8 }}>Filename</th>
                                            <th style={{ padding: 8 }}>Size</th>
                                            <th style={{ padding: 8 }}>Created</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {features.map(f => (
                                            <tr key={f.filename} style={{ borderTop: '1px solid var(--panel-border)' }}>
                                                <td style={{ padding: 8 }}>
                                                    <input
                                                        type="checkbox"
                                                        checked={selectedFeatures.includes(f.filename)}
                                                        onChange={() => toggleFeature(f.filename)}
                                                    />
                                                </td>
                                                <td style={{ padding: 8 }}>{f.filename}</td>
                                                <td style={{ padding: 8 }}>{f.size_mb} MB</td>
                                                <td style={{ padding: 8 }}>{new Date(f.created).toLocaleString()}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div style={{ flex: 1, padding: 20, background: 'var(--card-bg)', borderRadius: 8 }}>
                            <h4>Build Dataset</h4>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                                <label>
                                    Window Size (sec)
                                    <input type="number" value={buildParams.window_sec} onChange={e => setBuildParams({ ...buildParams, window_sec: e.target.value })} style={{ width: '100%' }} />
                                </label>
                                <label>
                                    Step Size (sec)
                                    <input type="number" value={buildParams.step_sec} onChange={e => setBuildParams({ ...buildParams, step_sec: e.target.value })} style={{ width: '100%' }} />
                                </label>
                                <label>
                                    Balance Method
                                    <select value={buildParams.balance_method} onChange={e => setBuildParams({ ...buildParams, balance_method: e.target.value })} style={{ width: '100%' }}>
                                        <option value="none">None</option>
                                        <option value="downsample">Downsample Majority</option>
                                        <option value="upsample">Upsample Minority</option>
                                    </select>
                                </label>
                                <button
                                    onClick={handleBuildDataset}
                                    disabled={loading || selectedFeatures.length === 0}
                                    style={{ marginTop: 10, padding: 10, background: 'var(--accent)', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}
                                >
                                    Create Dataset from {selectedFeatures.length} files
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* DATASETS TAB */}
            {activeTab === 'datasets' && (
                <div style={{ display: 'flex', gap: 20 }}>
                    <div style={{ flex: 2 }}>
                        <h4>Available Datasets</h4>
                        <div style={{ maxHeight: '60vh', overflow: 'auto' }}>
                            {datasets.map(d => (
                                <div
                                    key={d.id}
                                    onClick={() => setSelectedDataset(d.id)}
                                    style={{
                                        padding: 10,
                                        marginBottom: 10,
                                        border: `2px solid ${selectedDataset === d.id ? 'var(--accent)' : 'var(--panel-border)'}`,
                                        borderRadius: 6,
                                        cursor: 'pointer',
                                        background: selectedDataset === d.id ? 'rgba(var(--accent-rgb), 0.05)' : 'transparent'
                                    }}
                                >
                                    <div style={{ fontWeight: 'bold' }}>{d.id}</div>
                                    <div style={{ fontSize: '0.9em', color: 'var(--muted)' }}>
                                        Start: {d.train_size} train, {d.test_size} test samples.
                                        Features: {d.num_features}. Labels: {d.labels.join(', ')}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div style={{ flex: 1, padding: 20, background: 'var(--card-bg)', borderRadius: 8 }}>
                        <h4>Train Model</h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                            <label>
                                Algorithm
                                <select value={trainParams.model_type} onChange={e => setTrainParams({ ...trainParams, model_type: e.target.value })} style={{ width: '100%' }}>
                                    <option value="rf">Random Forest</option>
                                    <option value="lgbm">LightGBM</option>
                                    <option value="lstm">LSTM (Temporal)</option>
                                </select>
                            </label>
                            <label>
                                Aggregation
                                <select value={trainParams.aggregation} onChange={e => setTrainParams({ ...trainParams, aggregation: e.target.value })} style={{ width: '100%' }}>
                                    <option value="stats">Statistical (Mean/Std/etc)</option>
                                    <option value="flatten">Flatten Window</option>
                                </select>
                            </label>
                            <label>
                                Num Estimators
                                <input type="number" value={trainParams.n_estimators} onChange={e => setTrainParams({ ...trainParams, n_estimators: e.target.value })} style={{ width: '100%' }} />
                            </label>
                            <button
                                onClick={handleTrainModel}
                                disabled={loading || !selectedDataset}
                                style={{ marginTop: 10, padding: 10, background: 'var(--accent)', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}
                            >
                                Train Model on {selectedDataset || '...'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* MODELS TAB */}
            {activeTab === 'models' && (
                <div>
                    <h4>Trained Models</h4>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ background: 'var(--panel-bg)', textAlign: 'left' }}>
                                <th style={{ padding: 8 }}>Filename</th>
                                <th style={{ padding: 8 }}>Created</th>
                                <th style={{ padding: 8 }}>Accuracy</th>
                                <th style={{ padding: 8 }}>F1 Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {models.map(m => (
                                <tr key={m.filename} style={{ borderTop: '1px solid var(--panel-border)' }}>
                                    <td style={{ padding: 8 }}>{m.filename}</td>
                                    <td style={{ padding: 8 }}>{new Date(m.created).toLocaleString()}</td>
                                    <td style={{ padding: 8 }}>{(m.accuracy * 100).toFixed(1)}%</td>
                                    <td style={{ padding: 8 }}>{(m.f1_macro * 100).toFixed(1)}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}
