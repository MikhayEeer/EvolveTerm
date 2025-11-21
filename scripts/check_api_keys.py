#!/usr/bin/env python3
"""
Security check script to detect potential API keys in JSON files.
Intended to be used as a pre-commit hook or CI check.
"""

import json
import os
import subprocess
import sys
from typing import List, Dict, Any

# Whitelisted values that are definitely not real keys
ALLOWLIST = {
    "",
    "sk-",
    "REPLACE_ME",
    "YOUR_API_KEY",
    "YOUR_KEY",
    "api-key",
    "null",
    "None"
}

def get_tracked_json_files() -> List[str]:
    """Get all JSON files tracked by git."""
    try:
        # git ls-files lists all tracked files
        result = subprocess.run(
            ['git', 'ls-files'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        files = result.stdout.splitlines()
        return [f for f in files if f.endswith('.json') and os.path.exists(f)]
    except subprocess.CalledProcessError:
        print("Warning: Not a git repository or git command failed. Checking all local json files.")
        json_files = []
        for root, _, filenames in os.walk("."):
            if ".git" in root:
                continue
            for filename in filenames:
                if filename.endswith(".json"):
                    json_files.append(os.path.join(root, filename))
        return json_files

def is_potential_key(value: str) -> bool:
    """Heuristic to determine if a string looks like an API key."""
    if not isinstance(value, str):
        return False
    
    value = value.strip()
    
    if value in ALLOWLIST:
        return False
    
    # Common placeholders often contain these words
    if any(placeholder in value.upper() for placeholder in ["REPLACE", "TEMPLATE", "EXAMPLE", "YOUR_"]):
        return False

    # Check for specific patterns
    # OpenAI / DashScope keys often start with sk-
    if value.startswith("sk-"):
        # If it's just "sk-", it's likely a placeholder (already handled by allowlist, but double check)
        if len(value) <= 5: 
            return False
        return True
    
    # Generic long random string check
    # If it's long and has no spaces, it's suspicious for an api_key field
    if len(value) > 20 and " " not in value:
        return True
        
    return False

def check_json_content(data: Any, filepath: str) -> List[str]:
    """Recursively check JSON content for api_key fields."""
    issues = []

    def recursive_scan(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                current_path = f"{path}.{k}" if path else k
                
                # Check if the key itself suggests a secret
                if k == "api_key" or k == "secret_key" or k == "access_token":
                    if is_potential_key(v):
                        issues.append(f"Key: '{k}', Value: '{v[:4]}...{v[-4:] if len(v)>8 else ''}'")
                
                if isinstance(v, (dict, list)):
                    recursive_scan(v, current_path)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                recursive_scan(item, f"{path}[{i}]")

    recursive_scan(data)
    return issues

def main():
    print("🔍 Starting security check for API keys in JSON files...")
    
    files = get_tracked_json_files()
    if not files:
        print("No JSON files found to check.")
        sys.exit(0)

    print(f"Checking {len(files)} files...")
    
    failed_files = 0
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Allow comments in JSON if possible, but standard json lib doesn't support it.
                # We'll try standard load.
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Skip files that aren't valid JSON (or contain comments)
                    # print(f"  [WARN] Could not parse {filepath}, skipping.")
                    continue
            
            issues = check_json_content(data, filepath)
            
            if issues:
                print(f"\n❌ POTENTIAL LEAK IN: {filepath}")
                for issue in issues:
                    print(f"   - {issue}")
                failed_files += 1
                
        except Exception as e:
            print(f"  [ERR] Error reading {filepath}: {e}")

    if failed_files > 0:
        print(f"\n🚫 Security check FAILED. Found potential keys in {failed_files} file(s).")
        print("Action required: Replace real keys with placeholders (e.g., 'sk-', 'REPLACE_ME') or gitignore the file.")
        sys.exit(1)
    else:
        print("\n✅ Security check PASSED. No keys detected.")
        sys.exit(0)

if __name__ == "__main__":
    main()
