#!/usr/bin/env python3
"""
Commit Message Generator

This script analyzes git changes and generates descriptive commit messages
following conventional commit format.
"""

import subprocess
import sys
import re
from typing import List, Dict, Tuple, Optional


def run_git_command(args: List[str]) -> str:
    """Run a git command and return its output."""
    try:
        result = subprocess.run(
            ['git'] + args,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}", file=sys.stderr)
        return ""


def get_staged_changes() -> str:
    """Get the diff of staged changes."""
    return run_git_command(['diff', '--cached'])


def get_changed_files() -> List[str]:
    """Get list of files that have been changed."""
    output = run_git_command(['diff', '--cached', '--name-only'])
    return [f for f in output.split('\n') if f]


def analyze_file_changes() -> Dict[str, int]:
    """Analyze the types of files changed and count them."""
    files = get_changed_files()
    changes = {
        'python': 0,
        'javascript': 0,
        'jsx': 0,
        'css': 0,
        'html': 0,
        'json': 0,
        'markdown': 0,
        'other': 0
    }
    
    for file in files:
        if file.endswith('.py'):
            changes['python'] += 1
        elif file.endswith('.js'):
            changes['javascript'] += 1
        elif file.endswith('.jsx'):
            changes['jsx'] += 1
        elif file.endswith('.css'):
            changes['css'] += 1
        elif file.endswith('.html'):
            changes['html'] += 1
        elif file.endswith('.json'):
            changes['json'] += 1
        elif file.endswith('.md'):
            changes['markdown'] += 1
        else:
            changes['other'] += 1
    
    return changes


def determine_commit_type(diff: str, files: List[str]) -> str:
    """Determine the type of commit based on the changes."""
    diff_lower = diff.lower()
    
    # Check for new files
    if '+++' in diff and any('new file' in line.lower() for line in diff.split('\n')):
        return 'feat'
    
    # Check for deletions
    if '---' in diff and any('deleted file' in line.lower() for line in diff.split('\n')):
        return 'refactor'
    
    # Check for documentation changes
    if any(f.endswith('.md') or 'readme' in f.lower() for f in files):
        return 'docs'
    
    # Check for test changes
    if any('test' in f.lower() for f in files):
        return 'test'
    
    # Check for build/config changes
    if any(f in ['package.json', 'requirements.txt', 'setup.py', '.gitignore'] for f in files):
        return 'chore'
    
    # Check for bug fixes (common keywords)
    if any(keyword in diff_lower for keyword in ['fix', 'bug', 'issue', 'error', 'broken']):
        return 'fix'
    
    # Check for performance improvements
    if any(keyword in diff_lower for keyword in ['performance', 'optimize', 'faster', 'speed']):
        return 'perf'
    
    # Check for refactoring
    if any(keyword in diff_lower for keyword in ['refactor', 'restructure', 'reorganize']):
        return 'refactor'
    
    # Check for style changes
    if any(keyword in diff_lower for keyword in ['style', 'format', 'lint']):
        return 'style'
    
    # Default to feat for new functionality
    return 'feat'


def determine_scope(files: List[str]) -> Optional[str]:
    """Determine the scope of changes based on affected files."""
    if not files:
        return None
    
    # Check for backend changes
    backend_files = [f for f in files if 'backend' in f]
    if backend_files:
        # More specific backend scopes
        if any('server' in f for f in backend_files):
            return 'server'
        if any('script' in f for f in backend_files):
            return 'scripts'
        return 'backend'
    
    # Check for frontend changes
    frontend_files = [f for f in files if 'frontend' in f]
    if frontend_files:
        if any('component' in f for f in frontend_files):
            return 'components'
        if any('App.jsx' in f or 'main.jsx' in f for f in frontend_files):
            return 'app'
        return 'frontend'
    
    # Check for data changes
    if any('data' in f for f in files):
        return 'data'
    
    # Check for config changes
    if any(f in ['package.json', '.gitignore', 'requirements.txt'] for f in files):
        return 'config'
    
    return None


def generate_commit_description(diff: str, files: List[str]) -> str:
    """Generate a short description of the changes."""
    if not files:
        return "Update project files"
    
    file_changes = analyze_file_changes()
    
    # Count additions and deletions
    additions = len([line for line in diff.split('\n') if line.startswith('+')])
    deletions = len([line for line in diff.split('\n') if line.startswith('-')])
    
    # Describe based on what changed
    descriptions = []
    
    if file_changes['python'] > 0:
        descriptions.append(f"update Python code")
    if file_changes['javascript'] > 0 or file_changes['jsx'] > 0:
        descriptions.append(f"update React components")
    if file_changes['css'] > 0:
        descriptions.append(f"update styles")
    if file_changes['markdown'] > 0:
        descriptions.append(f"update documentation")
    if file_changes['json'] > 0:
        descriptions.append(f"update configuration")
    
    if not descriptions:
        descriptions.append(f"update {len(files)} file{'s' if len(files) > 1 else ''}")
    
    return descriptions[0] if descriptions else "update project"


def generate_commit_message() -> str:
    """Generate a complete commit message following conventional commits format."""
    # Check if there are any staged changes
    diff = get_staged_changes()
    if not diff:
        return "No staged changes to commit"
    
    files = get_changed_files()
    
    # Determine commit type and scope
    commit_type = determine_commit_type(diff, files)
    scope = determine_scope(files)
    description = generate_commit_description(diff, files)
    
    # Build the commit message
    if scope:
        message = f"{commit_type}({scope}): {description}"
    else:
        message = f"{commit_type}: {description}"
    
    return message


def main():
    """Main entry point."""
    message = generate_commit_message()
    print(message)
    
    # If --commit flag is provided, actually create the commit
    if '--commit' in sys.argv:
        if message != "No staged changes to commit":
            try:
                subprocess.run(['git', 'commit', '-m', message], check=True)
                print(f"\n✓ Committed with message: {message}")
            except subprocess.CalledProcessError as e:
                print(f"\nError creating commit: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print("\nNo changes to commit.")
            sys.exit(1)


if __name__ == '__main__':
    main()
