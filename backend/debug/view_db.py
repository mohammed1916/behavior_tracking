# python - <<'PY'
from backend.vector_store import STORE
import json
subs = STORE.list('subtasks')
print('TOTAL SUBTASKS:', len(subs))
# print summary
by_info = {}
for s in subs:
    m = s.get('metadata', {})
    info = m.get('subtask_info')
    key = info or s.get('id')
    by_info.setdefault(key, []).append({'id': s.get('id'), 'task_id': m.get('task_id'), 'created_at': m.get('created_at'), 'completed_in_time': m.get('completed_in_time'), 'completed_with_delay': m.get('completed_with_delay')})       
print(json.dumps(by_info, indent=2))
# PY
# TOTAL SUBTASKS: 4
# {
#   "Propeller": [
#     {
#       "id": "501f8180-b36e-4c87-afcf-d2637faf4650",
#       "task_id": "acaa9fe5-0181-400e-bf20-e11f5aed84d7",
#       "created_at": "2025-12-16T10:48:38.809538Z",
#       "completed_in_time": 0,
#       "completed_with_delay": 0
#     }
#   ],
#   "someone is making a model of a flying object with a small propeller": [
#     {
#       "id": "5fbfe70f-623b-4f20-b4c6-1e0c79c089b9",
#       "task_id": "acaa9fe5-0181-400e-bf20-e11f5aed84d7",
#       "created_at": "2025-12-16T11:21:45.356678Z",
#       "completed_in_time": 0,
#       "completed_with_delay": 4
#     },
#     {
#       "id": "ece1054b-2127-4ab5-ba52-3967f991894c",
#       "task_id": "61dda6fb-6998-4a4c-98bf-e5d640178683",
#       "created_at": "2025-12-17T05:16:49.147177Z",
#       "completed_in_time": 0,
#       "completed_with_delay": 1
#     },
#     {
#       "id": "52f4fd2a-70ff-45a0-84b3-135fbc845871",
#       "task_id": "55a6be9c-3459-4440-9ef8-34806996b7df",
#       "created_at": "2025-12-17T04:32:43.520592Z",
#       "completed_in_time": 0,
#       "completed_with_delay": 0
#     }
#   ]
# }