# Commit Message Generator

A Python utility that automatically generates descriptive commit messages following the [Conventional Commits](https://www.conventionalcommits.org/) specification.

## Features

- Analyzes staged git changes
- Determines appropriate commit type (feat, fix, docs, etc.)
- Identifies the scope of changes (backend, frontend, etc.)
- Generates descriptive commit messages automatically
- Follows Conventional Commits format

## Usage

### Generate a commit message

```bash
python backend/scripts/commit_message_generator.py
```

This will analyze your staged changes and print a suggested commit message.

### Generate and commit automatically

```bash
python backend/scripts/commit_message_generator.py --commit
```

This will generate a commit message and automatically create the commit.

## Commit Types

The generator automatically determines the commit type based on your changes:

- **feat**: New features or functionality
- **fix**: Bug fixes
- **docs**: Documentation changes
- **test**: Test changes
- **chore**: Build/config changes
- **perf**: Performance improvements
- **refactor**: Code refactoring
- **style**: Code style/formatting changes

## Scopes

The generator identifies the scope based on which part of the codebase changed:

- **server**: Backend server changes
- **scripts**: Script changes
- **backend**: General backend changes
- **components**: Frontend component changes
- **app**: Main app changes
- **frontend**: General frontend changes
- **config**: Configuration file changes
- **data**: Data-related changes

## Examples

```bash
# Stage your changes
git add backend/server.py

# Generate commit message
python backend/scripts/commit_message_generator.py
# Output: feat(server): update Python code

# Or commit directly
python backend/scripts/commit_message_generator.py --commit
```

## Requirements

- Python 3.6+
- Git repository with staged changes

## How it Works

1. Analyzes `git diff --cached` to see what's staged
2. Examines file types and names
3. Looks for keywords in the diff (fix, bug, performance, etc.)
4. Determines the most appropriate commit type and scope
5. Generates a concise description
6. Formats according to Conventional Commits: `type(scope): description`
