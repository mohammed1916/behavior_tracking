#!/usr/bin/env python3
"""
Test script for commit message generator

This script tests various scenarios for the commit message generator.
"""

import subprocess
import tempfile
import os
import sys

def run_command(cmd, cwd=None):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True, 
        text=True,
        cwd=cwd
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def test_no_changes():
    """Test when there are no staged changes."""
    print("\n=== Test 1: No staged changes ===")
    stdout, stderr, code = run_command(
        'python3 backend/scripts/commit_message_generator.py'
    )
    expected = "No staged changes to commit"
    if expected in stdout:
        print("✓ PASS: Correctly handles no staged changes")
        return True
    else:
        print(f"✗ FAIL: Expected '{expected}', got '{stdout}'")
        return False


def test_with_temp_repo():
    """Test with a temporary git repository."""
    print("\n=== Test 2: Create temporary repo with changes ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize a git repo
        run_command('git init', cwd=tmpdir)
        run_command('git config user.email "test@example.com"', cwd=tmpdir)
        run_command('git config user.name "Test User"', cwd=tmpdir)
        
        # Create a Python file
        test_file = os.path.join(tmpdir, 'test.py')
        with open(test_file, 'w') as f:
            f.write('# Test file\nprint("Hello world")\n')
        
        # Stage the file
        run_command('git add test.py', cwd=tmpdir)
        
        # Copy the generator script to the temp repo
        script_path = 'backend/scripts/commit_message_generator.py'
        with open(script_path, 'r') as src:
            script_content = src.read()
        
        temp_script = os.path.join(tmpdir, 'generator.py')
        with open(temp_script, 'w') as dst:
            dst.write(script_content)
        
        # Run the generator
        stdout, stderr, code = run_command('python3 generator.py', cwd=tmpdir)
        
        # Should generate a commit message
        if 'feat' in stdout or 'chore' in stdout:
            print(f"✓ PASS: Generated message: {stdout}")
            return True
        else:
            print(f"✗ FAIL: Expected commit message, got: {stdout}")
            if stderr:
                print(f"Error: {stderr}")
            return False


def test_import():
    """Test that the script can be imported without errors."""
    print("\n=== Test 3: Import test ===")
    try:
        # Try importing the functions
        sys.path.insert(0, 'backend/scripts')
        from commit_message_generator import (
            analyze_file_changes,
            determine_commit_type,
            determine_scope
        )
        print("✓ PASS: Script imports successfully")
        return True
    except Exception as e:
        print(f"✗ FAIL: Import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Commit Message Generator")
    print("=" * 50)
    
    results = []
    results.append(test_no_changes())
    results.append(test_with_temp_repo())
    results.append(test_import())
    
    print("\n" + "=" * 50)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
