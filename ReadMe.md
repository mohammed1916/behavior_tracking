Running

## 1.1 Frontend

cd frontend;npm run dev

## 1.2 Backend

cd backend;python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload

### Git Bash:

cd backend && python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload

### TaskKill all 8001 process: (git Bash):

cmd /c "for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001') do taskkill /PID %%a /F"

or

netstat -ano | grep LISTENING | grep :8001 | awk '{print $5}' | xargs -r taskkill //PID //F
