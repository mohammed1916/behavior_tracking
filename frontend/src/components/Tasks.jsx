import React, { useState, useEffect } from 'react';

function Tasks({}) {
  const [tasks, setTasks] = useState([]);
  const [newTaskName, setNewTaskName] = useState('');
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [editing, setEditing] = useState(null);
  const [editName, setEditName] = useState('');

  const fetchTasks = async () => {
    setLoading(true);
    try {
      const resp = await fetch('http://localhost:8001/backend/tasks');
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      setTasks(data);
    } catch (e) {
      console.error('Failed to fetch tasks', e);
    } finally {
      setLoading(false);
    }
  };

  const createTask = async () => {
    if (!newTaskName.trim()) return alert('Enter task name');
    setCreating(true);
    try {
      const form = new FormData();
      form.append('name', newTaskName);
      const resp = await fetch('http://localhost:8001/backend/tasks', { method: 'POST', body: form });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      alert(`Task created: ${data.task_id}`);
      setNewTaskName('');
      fetchTasks(); // Refresh list
    } catch (e) {
      alert('Failed to create task: ' + e.message);
    } finally {
      setCreating(false);
    }
  };

  const updateTask = async (id, name) => {
    try {
      const form = new FormData();
      form.append('name', name);
      const resp = await fetch(`http://localhost:8001/backend/tasks/${id}`, { method: 'PUT', body: form });
      if (!resp.ok) throw new Error(await resp.text());
      fetchTasks();
      setEditing(null);
    } catch (e) {
      alert('Failed to update task: ' + e.message);
    }
  };

  const deleteTask = async (id) => {
    if (!confirm('Delete this task? This will delete all related assignments.')) return;
    try {
      const resp = await fetch(`http://localhost:8001/backend/tasks/${id}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error(await resp.text());
      fetchTasks();
    } catch (e) {
      alert('Failed to delete task: ' + e.message);
    }
  };

  useEffect(() => {
    fetchTasks();
  }, []);

  return (
    <div style={{ padding: 20, backgroundColor: 'var(--surface)', borderRadius: 8, boxShadow: 'var(--panel-shadow)' }}>
      <h3 style={{ marginTop: 0 }}>Tasks Management</h3>
      <div style={{ marginBottom: 20, padding: 16, backgroundColor: 'var(--card-bg)', borderRadius: 6, border: '1px solid var(--panel-border)' }}>
        <h4 style={{ marginTop: 0 }}>Create New Task</h4>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <label>
            Task Name:
            <input 
              value={newTaskName} 
              onChange={(e) => setNewTaskName(e.target.value)} 
              placeholder="e.g. Assembly Line Work" 
              style={{ width: '100%', padding: 8, border: '1px solid var(--panel-border)', borderRadius: 4, backgroundColor: 'var(--surface)', color: 'var(--text)' }}
            />
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
                  <th style={{ border: '1px solid var(--panel-border)', padding: 12 }}>Task Name</th>
                  <th style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {tasks.map((t, index) => (
                  <tr key={t.id} style={{ backgroundColor: index % 2 === 0 ? 'var(--surface)' : 'var(--card-bg)' }}>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12 }}>
                      {editing === t.id ? (
                        <input 
                          value={editName} 
                          onChange={(e) => setEditName(e.target.value)} 
                          style={{ width: '100%', padding: 8, border: '1px solid var(--panel-border)', borderRadius: 4, backgroundColor: 'var(--surface)', color: 'var(--text)' }}
                        />
                      ) : (
                        t.name
                      )}
                    </td>
                    <td style={{ border: '1px solid var(--panel-border)', padding: 12, textAlign: 'center' }}>
                      {editing === t.id ? (
                        <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
                          <button 
                            onClick={() => updateTask(t.id, editName)} 
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
                            onClick={() => { setEditing(t.id); setEditName(t.name); }} 
                            style={{ padding: 6, backgroundColor: 'var(--accent)', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}
                          >
                            Edit
                          </button>
                          <button 
                            onClick={() => deleteTask(t.id)} 
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

export default Tasks;