import React, { useState, useEffect } from 'react';

function Tasks({ onTaskSelect }) {
  const [tasks, setTasks] = useState([]);
  const [newTaskText, setNewTaskText] = useState('');
  const [newExpectedDuration, setNewExpectedDuration] = useState('');
  const [useLLMForDuration, setUseLLMForDuration] = useState(false);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState(null);
  const [editText, setEditText] = useState('');
  const [editDuration, setEditDuration] = useState('');

  const fetchTasks = async () => {
    setLoading(true);
    try {
      const resp = await fetch('http://localhost:8001/backend/assignments');
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      setTasks(data.map(t => ({ ...t, selected: false, completed: false })));
    } catch (e) {
      console.error('Failed to fetch tasks', e);
    } finally {
      setLoading(false);
    }
  };

  const createTask = async () => {
    if (!newTaskText.trim()) return alert('Enter task text');
    setCreating(true);
    try {
      const form = new FormData();
      form.append('task_text', newTaskText);
      if (newExpectedDuration) form.append('expected_duration_sec', newExpectedDuration);
      form.append('use_llm_for_duration', useLLMForDuration.toString());
      const resp = await fetch('http://localhost:8001/backend/assign_task', { method: 'POST', body: form });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      alert(`Task created: ${data.assignment_id}`);
      setNewTaskText('');
      setNewExpectedDuration('');
      setUseLLMForDuration(false);
      fetchTasks(); // Refresh list
    } catch (e) {
      alert('Failed to create task: ' + e.message);
    } finally {
      setCreating(false);
    }
  };

  const updateTask = async (id, taskText, expectedDuration) => {
    try {
      const form = new FormData();
      form.append('task_text', taskText);
      if (expectedDuration) form.append('expected_duration_sec', expectedDuration);
      const resp = await fetch(`http://localhost:8001/backend/assignments/${id}`, { method: 'PUT', body: form });
      if (!resp.ok) throw new Error(await resp.text());
      fetchTasks();
      setEditing(null);
    } catch (e) {
      alert('Failed to update task: ' + e.message);
    }
  };

  const deleteTask = async (id) => {
    if (!confirm('Delete this task?')) return;
    try {
      const resp = await fetch(`http://localhost:8001/backend/assignments/${id}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error(await resp.text());
      fetchTasks();
    } catch (e) {
      alert('Failed to delete task: ' + e.message);
    }
  };

  const handleSelect = (id, selected) => {
    setTasks(tasks.map(t => t.id === id ? { ...t, selected } : t));
  };

  const handleCompleted = (id, completed) => {
    setTasks(tasks.map(t => t.id === id ? { ...t, completed } : t));
  };

  useEffect(() => {
    onTaskSelect(tasks.filter(t => t.selected).map(t => ({ id: t.id, completed: t.completed })));
  }, [tasks, onTaskSelect]);

  return (
    <div style={{ padding: 20, backgroundColor: 'var(--surface)', borderRadius: 8, boxShadow: 'var(--panel-shadow)' }}>
      <h3 style={{ marginTop: 0 }}>Tasks Management</h3>
      <div style={{ marginBottom: 20, padding: 16, backgroundColor: 'var(--card-bg)', borderRadius: 6, border: '1px solid var(--panel-border)' }}>
        <h4 style={{ marginTop: 0 }}>Create New Task</h4>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <label>
            Task Description:
            <textarea 
              value={newTaskText} 
              onChange={(e) => setNewTaskText(e.target.value)} 
              placeholder="Describe the task in detail" 
              rows={3} 
              style={{ width: '100%', padding: 8, border: '1px solid var(--panel-border)', borderRadius: 4, backgroundColor: 'var(--surface)', color: 'var(--text)' }}
            />
          </label>
          <label>
            Expected Duration (seconds, optional):
            <input 
              type="number" 
              value={newExpectedDuration} 
              onChange={(e) => setNewExpectedDuration(e.target.value)} 
              style={{ width: '100%', padding: 8, border: '1px solid var(--panel-border)', borderRadius: 4, backgroundColor: 'var(--surface)', color: 'var(--text)' }}
            />
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input 
              type="checkbox" 
              checked={useLLMForDuration} 
              onChange={(e) => setUseLLMForDuration(e.target.checked)} 
            />
            Use LLM to estimate duration if not provided
          </label>
          <button 
            onClick={createTask} 
            disabled={creating} 
            style={{ padding: 10, backgroundColor: 'var(--accent)', color: 'white', border: 'none', borderRadius: 4, cursor: creating ? 'not-allowed' : 'pointer' }}
          >
            {creating ? 'Creating...' : 'Create Task'}
          </button>
        </div>
      </div>
      <div>
        <h4>Existing Tasks</h4>
        {loading ? <p>Loading tasks...</p> : tasks.length === 0 ? <p>No tasks created yet. Create your first task above.</p> : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', backgroundColor: 'var(--card-bg)', borderRadius: 6, overflow: 'hidden' }}>
              <thead>
                <tr style={{ backgroundColor: 'var(--accent-ghost)' }}>
                  <th style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>Select</th>
                  <th style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>Completed</th>
                  <th style={{ border: '1px solid var(--panel-border)', padding: 12 }}>Task Description</th>
                  <th style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>Duration (s)</th>
                  <th style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {tasks.map((t, index) => (
                  <tr key={t.id} style={{ backgroundColor: index % 2 === 0 ? 'var(--surface)' : 'var(--card-bg)', ':hover': { backgroundColor: 'var(--accent-ghost)' } }}>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>
                      <input type="checkbox" checked={t.selected} onChange={(e) => handleSelect(t.id, e.target.checked)} />
                    </td>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>
                      <input type="checkbox" checked={t.completed} onChange={(e) => handleCompleted(t.id, e.target.checked)} />
                    </td>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12 }}>
                      {editing === t.id ? (
                        <textarea 
                          value={editText} 
                          onChange={(e) => setEditText(e.target.value)} 
                          rows={2} 
                          style={{ width: '100%', padding: 8, border: '1px solid var(--panel-border)', borderRadius: 4, backgroundColor: 'var(--surface)', color: 'var(--text)' }}
                        />
                      ) : (
                        <div style={{ wordWrap: 'break-word' }}>{t.task_text}</div>
                      )}
                    </td>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>
                      {editing === t.id ? (
                        <input 
                          type="number" 
                          value={editDuration} 
                          onChange={(e) => setEditDuration(e.target.value)} 
                          style={{ width: '100%', padding: 8, border: '1px solid var(--panel-border)', borderRadius: 4, backgroundColor: 'var(--surface)', color: 'var(--text)' }}
                        />
                      ) : (
                        t.expected_duration_sec
                      )}
                    </td>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>
                      {editing === t.id ? (
                        <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
                          <button 
                            onClick={() => updateTask(t.id, editText, editDuration)} 
                            style={{ padding: 6, backgroundColor: 'var(--accent)', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}
                          >
                            Save
                          </button>
                          <button 
                            onClick={() => setEditing(null)} 
                            style={{ padding: 6, backgroundColor: '#6b7280', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
                          <button 
                            onClick={() => { setEditing(t.id); setEditText(t.task_text); setEditDuration(t.expected_duration_sec); }} 
                            style={{ padding: 6, backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}
                          >
                            Edit
                          </button>
                          <button 
                            onClick={() => deleteTask(t.id)} 
                            style={{ padding: 6, backgroundColor: '#ef4444', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}
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

export default Tasks;