# Quick Start: Commit Message Generator

## Installation
No installation required! The script is ready to use.

## Basic Usage

### 1. Preview Mode (Recommended)
Stage your changes and preview the generated commit message:

```bash
git add <your-files>
python backend/scripts/commit_message_generator.py
```

### 2. Auto-Commit Mode
Stage your changes and commit automatically:

```bash
git add <your-files>
python backend/scripts/commit_message_generator.py --commit
```

## Quick Examples

### Example 1: Adding a new feature
```bash
# Add a new Python function to backend
git add backend/server.py
python backend/scripts/commit_message_generator.py
# Output: feat(server): update Python code

# Commit it
python backend/scripts/commit_message_generator.py --commit
```

### Example 2: Updating documentation
```bash
# Update README
git add ReadMe.md
python backend/scripts/commit_message_generator.py
# Output: docs: update documentation
```

### Example 3: Fixing a bug
```bash
# Fix a bug in React component
git add frontend/src/components/FileUpload.jsx
python backend/scripts/commit_message_generator.py
# Output: fix(components): update React components
```

## What It Does

The generator analyzes your staged changes and automatically:
1. ✅ Determines commit type (feat, fix, docs, etc.)
2. ✅ Identifies the scope (backend, frontend, scripts, etc.)
3. ✅ Creates a descriptive message
4. ✅ Follows Conventional Commits format

## Commit Types You'll See

- **feat**: New features
- **fix**: Bug fixes
- **docs**: Documentation changes
- **test**: Test changes
- **chore**: Config/build changes
- **refactor**: Code restructuring
- **perf**: Performance improvements
- **style**: Formatting changes

## Tips

💡 **Stage related changes together** for better commit messages

💡 **Review the generated message** before using --commit

💡 **Use preview mode first** to see what will be committed

## Need Help?

See full documentation: [backend/scripts/README_commit_generator.md](backend/scripts/README_commit_generator.md)

## Run Tests

```bash
python backend/scripts/test_commit_generator.py
```

---

**That's it!** You're ready to create better commit messages! 🎉
