#!/usr/bin/env python3
"""Simple test for yaml_schema module without dependencies."""

import sys
from pathlib import Path

# Direct import without going through __init__
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import only yaml_schema module (no pipeline dependencies)
import yaml
from evolve_term.yaml_schema import validate_yaml_content


def test_valid_extract():
    """Test valid extract YAML."""
    content = {
        "source_path": "test.c",
        "task": "extract",
        "command": "evolveterm extract",
        "pmt_ver": "v2",
        "model": "qwen3-max",
        "time": "2026-01-09",
        "loops_count": -1,
        "loops_depth": -1,
        "loops_ids": 2,
        "loops": [
            {"id": 1, "code": "for(j=0;j<m;j++) { x++; }"},
            {"id": 2, "code": "while(i<n) { LOOP1; i++; }"}
        ]
    }
    
    result = validate_yaml_content(content, "extract")
    print(f"✓ Test: Valid Extract YAML")
    print(f"  Result: {'PASS' if result.valid else 'FAIL'}")
    if not result.valid:
        for err in result.errors:
            print(f"    Error: {err}")
    assert result.valid, "Should be valid"
    print()


def test_invalid_loop_order():
    """Test invalid loop ID order."""
    content = {
        "source_path": "test.c",
        "task": "extract",
        "command": "evolveterm extract",
        "pmt_ver": "v2",
        "model": "qwen3-max",
        "time": "2026-01-09",
        "loops_count": -1,
        "loops_depth": -1,
        "loops_ids": 2,
        "loops": [
            {"id": 2, "code": "while(i<n) { i++; }"},  # Wrong!
            {"id": 1, "code": "for(j=0;j<m;j++) { x++; }"}
        ]
    }
    
    result = validate_yaml_content(content, "extract")
    print(f"✓ Test: Invalid Loop ID Order")
    print(f"  Result: {'PASS' if not result.valid else 'FAIL'}")
    if result.errors:
        print(f"  Detected errors:")
        for err in result.errors:
            print(f"    - {err}")
    assert not result.valid, "Should be invalid"
    print()


def test_forward_reference():
    """Test forward reference detection."""
    content = {
        "source_path": "test.c",
        "task": "extract",
        "command": "evolveterm extract",
        "pmt_ver": "v2",
        "model": "qwen3-max",
        "time": "2026-01-09",
        "loops_count": -1,
        "loops_depth": -1,
        "loops_ids": 2,
        "loops": [
            {"id": 1, "code": "while(i<n) { LOOP2; i++; }"},  # Forward ref!
            {"id": 2, "code": "for(j=0;j<m;j++) { x++; }"}
        ]
    }
    
    result = validate_yaml_content(content, "extract")
    print(f"✓ Test: Forward Reference Detection")
    print(f"  Result: {'PASS' if not result.valid else 'FAIL'}")
    if result.errors:
        print(f"  Detected errors:")
        for err in result.errors:
            print(f"    - {err}")
    assert not result.valid, "Should detect forward reference"
    print()


def test_missing_fields():
    """Test missing required fields."""
    content = {
        "source_path": "test.c",
        "task": "extract",
        # Missing many required fields
    }
    
    result = validate_yaml_content(content, "extract")
    print(f"✓ Test: Missing Required Fields")
    print(f"  Result: {'PASS' if not result.valid else 'FAIL'}")
    print(f"  Missing fields count: {len(result.errors)}")
    assert not result.valid, "Should detect missing fields"
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("YAML Schema Validation - Simple Tests")
    print("=" * 60)
    print()
    
    try:
        test_valid_extract()
        test_invalid_loop_order()
        test_forward_reference()
        test_missing_fields()
        
        print("=" * 60)
        print("✓ All tests PASSED!")
        print("=" * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
