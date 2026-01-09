#!/usr/bin/env python3
"""Test script for YAML schema validation.

Usage:
    python test_yaml_validation.py
"""

from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evolve_term.yaml_schema import (
    validate_yaml_content,
    validate_yaml_file,
    ValidationResult,
)


def test_valid_extract_yaml():
    """Test validation of a valid extract YAML."""
    content = {
        "source_path": "test.c",
        "task": "extract",
        "command": "evolveterm extract --input test.c",
        "pmt_ver": "pmt_yamlv2",
        "model": "qwen3-max",
        "time": "2026-01-09T16:00",
        "loops_count": -1,
        "loops_depth": -1,
        "loops_ids": 2,
        "loops": [
            {"id": 1, "code": "for(j=0;j<m;j++) { x++; }"},
            {"id": 2, "code": "while(i<n) { LOOP1; i++; }"}
        ]
    }
    
    result = validate_yaml_content(content, "extract", strict=False)
    print(f"✓ Valid Extract YAML: {result.valid}")
    assert result.valid
    print(f"  Warnings: {len(result.warnings)}")


def test_invalid_loop_id_order():
    """Test detection of invalid loop ID order."""
    content = {
        "source_path": "test.c",
        "task": "extract",
        "command": "evolveterm extract --input test.c",
        "pmt_ver": "pmt_yamlv2",
        "model": "qwen3-max",
        "time": "2026-01-09T16:00",
        "loops_count": -1,
        "loops_depth": -1,
        "loops_ids": 2,
        "loops": [
            {"id": 2, "code": "while(i<n) { i++; }"},  # Wrong order!
            {"id": 1, "code": "for(j=0;j<m;j++) { x++; }"}
        ]
    }
    
    result = validate_yaml_content(content, "extract", strict=False)
    print(f"✓ Invalid Loop ID Order Detected: {not result.valid}")
    assert not result.valid
    print(f"  Errors: {result.errors}")


def test_forward_reference():
    """Test detection of forward references in nested loops."""
    content = {
        "source_path": "test.c",
        "task": "extract",
        "command": "evolveterm extract --input test.c",
        "pmt_ver": "pmt_yamlv2",
        "model": "qwen3-max",
        "time": "2026-01-09T16:00",
        "loops_count": -1,
        "loops_depth": -1,
        "loops_ids": 2,
        "loops": [
            {"id": 1, "code": "while(i<n) { LOOP2; i++; }"},  # Forward reference!
            {"id": 2, "code": "for(j=0;j<m;j++) { x++; }"}
        ]
    }
    
    result = validate_yaml_content(content, "extract", strict=False)
    print(f"✓ Forward Reference Detected: {not result.valid}")
    assert not result.valid
    print(f"  Errors: {result.errors}")


def test_valid_ranking_yaml():
    """Test validation of a valid ranking YAML."""
    content = {
        "source_file": "test_inv.yml",
        "source_path": "test.c",
        "task": "ranking_inference",
        "command": "evolveterm ranking --input test.yml",
        "pmt_ver": "template",
        "model": "qwen3-max",
        "time": "2026-01-09T16:00",
        "has_extract": True,
        "has_invariants": True,
        "ranking_results": [
            {
                "loop_id": 1,
                "code": "for(j=0;j<m;j++) { x++; }",
                "invariants": ["j >= 0", "j <= m"],
                "template_type": "lnested",
                "template_depth": 1,
                "explanation": "Simple linear loop"
            },
            {
                "loop_id": 2,
                "code": "while(i<n) { LOOP1; i++; }",
                "invariants": ["i >= 0", "i <= n"],
                "template_type": "lnested",
                "template_depth": 2,
                "explanation": "Nested loop structure"
            }
        ]
    }
    
    result = validate_yaml_content(content, "ranking", strict=False)
    print(f"✓ Valid Ranking YAML: {result.valid}")
    assert result.valid


def test_missing_required_field():
    """Test detection of missing required fields."""
    content = {
        "source_path": "test.c",
        "task": "extract",
        # Missing: command, pmt_ver, model, time, loops_count, loops_depth, loops_ids, loops
    }
    
    result = validate_yaml_content(content, "extract", strict=False)
    print(f"✓ Missing Fields Detected: {not result.valid}")
    assert not result.valid
    print(f"  Missing fields: {len(result.errors)}")


def test_valid_nested_loops():
    """Test validation of correctly nested loops."""
    content = {
        "source_path": "test.c",
        "task": "extract",
        "command": "evolveterm extract --input test.c",
        "pmt_ver": "pmt_yamlv2",
        "model": "qwen3-max",
        "time": "2026-01-09T16:00",
        "loops_count": -1,
        "loops_depth": -1,
        "loops_ids": 3,
        "loops": [
            {"id": 1, "code": "for(k=0;k<p;k++) { z++; }"},
            {"id": 2, "code": "for(j=0;j<m;j++) { LOOP1; x++; }"},
            {"id": 3, "code": "while(i<n) { LOOP2; i++; }"}
        ]
    }
    
    result = validate_yaml_content(content, "extract", strict=False)
    print(f"✓ Valid Nested Loops: {result.valid}")
    assert result.valid
    print(f"  Loop 3 references LOOP2 (which is loop 2, already defined)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("EvolveTerm YAML Schema Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Valid Extract YAML", test_valid_extract_yaml),
        ("Invalid Loop ID Order", test_invalid_loop_id_order),
        ("Forward Reference Detection", test_forward_reference),
        ("Valid Ranking YAML", test_valid_ranking_yaml),
        ("Missing Required Fields", test_missing_required_field),
        ("Valid Nested Loops", test_valid_nested_loops),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"Test: {test_name}")
        try:
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            print()
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
