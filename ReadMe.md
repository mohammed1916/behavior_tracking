Running

## 1.1 Frontend

cd frontend;npm run dev

## 1.2 Backend

python -m uvicorn backend.server:app --host 0.0.0.0 --port 8001 --reload

### Git Bash:

python -m uvicorn backend.server:app --host 0.0.0.0 --port 8001 --reload

Common pitfall :-

Do not use the following, becase the following might not identify backend module as it is already in backend: (scipts inside it have backend `<dot>` import):

cd backend && python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload

### TaskKill all 8001 process: (git Bash):

cmd /c "for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001') do taskkill /PID %%a /F"

or

netstat -ano | grep LISTENING | grep :8001 | awk '{print $5}' | xargs -r taskkill //PID //F
