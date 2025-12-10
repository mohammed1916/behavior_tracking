# Behavior Tracking Application

A video behavior analysis application with AI-powered tracking and classification.

## 1. Running

### 1.1 Frontend

```bash
cd frontend && npm run dev
```

### 1.2 Backend

```bash
cd backend && python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

## 2. Developer Tools

### 2.1 Commit Message Generator

Automatically generate conventional commit messages based on your staged changes:

```bash
# Generate commit message
python backend/scripts/commit_message_generator.py

# Generate and commit automatically
python backend/scripts/commit_message_generator.py --commit
```

See [backend/scripts/README_commit_generator.md](backend/scripts/README_commit_generator.md) for more details.
