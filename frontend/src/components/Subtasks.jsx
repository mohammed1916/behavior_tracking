import React, { useState, useEffect } from 'react';

function Subtasks({ refreshTrigger }) {
  const [subtasks, setSubtasks] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [newSubtaskInfo, setNewSubtaskInfo] = useState('');
  const [newDuration, setNewDuration] = useState('');
  const [selectedTaskId, setSelectedTaskId] = useState('');
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState(null);
  const [editInfo, setEditInfo] = useState('');
  const [editDuration, setEditDuration] = useState('');

  const fetchSubtasks = async () => {
    setLoading(true);
    try {
      const resp = await fetch('http://localhost:8001/backend/subtasks');
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      setSubtasks(data);
    } catch (e) {
      console.error('Failed to fetch subtasks', e);
    } finally {
      setLoading(false);
    }
  };

  const fetchTasks = async () => {
    try {
      const resp = await fetch('http://localhost:8001/backend/tasks');
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      setTasks(data);
    } catch (e) {
      console.error('Failed to fetch tasks', e);
    }
  };

  const createSubtask = async () => {
    if (!selectedTaskId || !newSubtaskInfo.trim() || !newDuration.trim()) return alert('Select task and enter duration');
    setCreating(true);
    try {
      const form = new FormData();
      form.append('task_id', selectedTaskId);
      form.append('subtask_info', newSubtaskInfo);
      form.append('duration_sec', newDuration);
      const resp = await fetch('http://localhost:8001/backend/subtasks', { method: 'POST', body: form });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      alert(`Subtask created: ${data.subtask_id}`);
      setNewSubtaskInfo('');
      setNewDuration('');
      setSelectedTaskId('');
      fetchSubtasks(); // Refresh list
    } catch (e) {
      alert('Failed to create subtask: ' + e.message);
    } finally {
      setCreating(false);
    }
  };

  const updateSubtask = async (id, info, duration) => {
    try {
      const form = new FormData();
      form.append('subtask_info', info);
      form.append('duration_sec', duration);
      const resp = await fetch(`http://localhost:8001/backend/subtasks/${id}`, { method: 'PUT', body: form });
      if (!resp.ok) throw new Error(await resp.text());
      fetchSubtasks();
      setEditing(null);
    } catch (e) {
      alert('Failed to update subtask: ' + e.message);
    }
  };

  const deleteSubtask = async (id) => {
    if (!confirm('Delete this subtask?')) return;
    try {
      const resp = await fetch(`http://localhost:8001/backend/subtasks/${id}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error(await resp.text());
      fetchSubtasks();
    } catch (e) {
      alert('Failed to delete subtask: ' + e.message);
    }
  };

  

  useEffect(() => {
    fetchSubtasks();
    fetchTasks();
  }, []);

  useEffect(() => {
    if (refreshTrigger > 0) {
      fetchSubtasks();
    }
  }, [refreshTrigger]);

  // selection feature removed: no callback to parent

  return (
    <div style={{ padding: 20, backgroundColor: 'var(--surface)', borderRadius: 8, boxShadow: 'var(--panel-shadow)' }}>
      <h3 style={{ marginTop: 0 }}>Subtasks Management</h3>
      <div style={{ marginBottom: 20, padding: 16, backgroundColor: 'var(--card-bg)', borderRadius: 6, border: '1px solid var(--panel-border)' }}>
        <h4 style={{ marginTop: 0 }}>Create New Subtask</h4>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <label>
            Select Task:
            <select 
              value={selectedTaskId} 
              onChange={(e) => setSelectedTaskId(e.target.value)} 
              style={{ width: '100%', padding: 8, border: '1px solid var(--panel-border)', borderRadius: 4, backgroundColor: 'var(--surface)', color: 'var(--text)' }}
            >
              <option value="">Choose a task...</option>
              {tasks.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
            </select>
          </label>
          <label>
            Sub Task Info:
            <textarea 
              value={newSubtaskInfo} 
              onChange={(e) => setNewSubtaskInfo(e.target.value)} 
              placeholder="Additional details for this subtask" 
              rows={2} 
              style={{ width: '100%', padding: 8, border: '1px solid var(--panel-border)', borderRadius: 4, backgroundColor: 'var(--surface)', color: 'var(--text)' }}
            />
          </label>
          <label>
            Duration (seconds):
            <input 
              type="number" 
              value={newDuration} 
              onChange={(e) => setNewDuration(e.target.value)} 
              placeholder="e.g. 300" 
              required
              style={{ width: '100%', padding: 8, border: '1px solid var(--panel-border)', borderRadius: 4, backgroundColor: 'var(--surface)', color: 'var(--text)' }}
            />
          </label>
          <button 
            onClick={createSubtask} 
            disabled={creating} 
            style={{ padding: 10, backgroundColor: 'var(--accent)', color: 'white', border: 'none', borderRadius: 4, cursor: creating ? 'not-allowed' : 'pointer' }}
          >
            {creating ? 'Creating...' : 'Create Subtask'}
          </button>
        </div>
      </div>
      <div>
        <h4>Existing Subtasks</h4>
        {loading ? <p>Loading subtasks...</p> : subtasks.length === 0 ? <p>No subtasks created yet. Create your first subtask above.</p> : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', backgroundColor: 'var(--card-bg)', borderRadius: 6, overflow: 'hidden' }}>
                <thead>
                  <tr style={{ backgroundColor: 'var(--accent-ghost)' }}>
                    <th style={{ border: '1px solid var(--panel-border)', padding: 12 }}>Task</th>
                    <th style={{ border: '1px solid var(--panel-border)', padding: 12 }}>Sub Task Info</th>
                    <th style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>Duration (s)</th>
                    <th style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>Completed In Time</th>
                    <th style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>Completed With Delay</th>
                    <th style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>Actions</th>
                  </tr>
                </thead>
              <tbody>
                {subtasks.map((a, index) => (
                  <tr key={a.id} style={{ backgroundColor: index % 2 === 0 ? 'var(--surface)' : 'var(--card-bg)' }}>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12 }}>
                      {a.task_name}
                    </td>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12 }}>
                      {editing === a.id ? (
                        <textarea 
                          value={editInfo} 
                          onChange={(e) => setEditInfo(e.target.value)} 
                          rows={2} 
                          style={{ width: '100%', padding: 8, border: '1px solid var(--panel-border)', borderRadius: 4, backgroundColor: 'var(--surface)', color: 'var(--text)' }}
                        />
                      ) : (
                        <div style={{ wordWrap: 'break-word' }}>{a.subtask_info || 'N/A'}</div>
                      )}
                    </td>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>
                      {editing === a.id ? (
                        <input 
                          type="number" 
                          value={editDuration} 
                          onChange={(e) => setEditDuration(e.target.value)} 
                          style={{ width: '100%', padding: 8, border: '1px solid var(--panel-border)', borderRadius: 4, backgroundColor: 'var(--surface)', color: 'var(--text)' }}
                        />
                      ) : (
                        a.duration_sec
                      )}
                    </td>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>
                      {a.completed_in_time}
                    </td>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>
                      {a.completed_with_delay}
                    </td>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>
                      {editing === a.id ? (
                        <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
                          <button 
                            onClick={() => updateSubtask(a.id, editInfo, editDuration)} 
                            style={{ padding: 6, backgroundColor: 'var(--accent)', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}
                          >
                            Save
                          </button>
                          <button 
                            onClick={() => setEditing(null)} 
                            style={{ padding: 6, backgroundColor: 'var(--muted)', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
                          <button 
                            onClick={() => { setEditing(a.id); setEditInfo(a.subtask_info || ''); setEditDuration(a.duration_sec); }} 
                            style={{ padding: 6, backgroundColor: 'var(--accent)', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}
                          >
                            Edit
                          </button>
                          <button 
                            onClick={() => deleteSubtask(a.id)} 
                            style={{ padding: 6, backgroundColor: 'var(--danger, #ef4444)', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}
                          >
                            Delete
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default Subtasks;