"""Unified YAML schema validation for EvolveTerm pipeline.

This module provides comprehensive validation for all YAML formats used across
extract, invariant, ranking, and verification modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import re

import yaml


# ============================================================================
# Schema Definitions
# ============================================================================

@dataclass
class FieldSpec:
    """Specification for a YAML field."""
    required: bool = True
    field_type: Optional[type] = None
    allowed_values: Optional[Set[Any]] = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None


# Base fields shared across all YAML types
_BASE_METADATA_FIELDS = {
    "source_path": FieldSpec(required=True, field_type=str),
    "task": FieldSpec(required=True, field_type=str),
    "command": FieldSpec(required=True, field_type=str),
    "pmt_ver": FieldSpec(required=True, field_type=str),
    "model": FieldSpec(required=True, field_type=str),
    "time": FieldSpec(required=True, field_type=str),
}

# Extract YAML schema
EXTRACT_SCHEMA = {
    **_BASE_METADATA_FIELDS,
    "loops_count": FieldSpec(required=True, field_type=int),
    "loops_depth": FieldSpec(required=True, field_type=int),
    "loops_ids": FieldSpec(required=True, field_type=int, min_value=0),
    "loops": FieldSpec(required=True, field_type=list),
}

# Invariant YAML schema
INVARIANT_SCHEMA = {
    **_BASE_METADATA_FIELDS,
    "source_file": FieldSpec(required=True, field_type=str),
    "has_extract": FieldSpec(required=True, field_type=bool),
    "invariants_result": FieldSpec(required=True, field_type=list),
}

# Ranking YAML schema
RANKING_SCHEMA = {
    **_BASE_METADATA_FIELDS,
    "source_file": FieldSpec(required=True, field_type=str),
    "has_extract": FieldSpec(required=True, field_type=bool),
    "has_invariants": FieldSpec(required=True, field_type=bool),
    "ranking_results": FieldSpec(required=True, field_type=list),
}

# Feature YAML schema
FEATURE_SCHEMA = {
    "source_path": FieldSpec(required=True, field_type=str),
    "language": FieldSpec(required=True, field_type=str),
    "program_type": FieldSpec(required=True, field_type=str),
    "recur_type": FieldSpec(required=True, field_type=str),
    "loop_type": FieldSpec(required=True, field_type=str),
    "loops_count": FieldSpec(required=True, field_type=int),
    "loops_depth": FieldSpec(required=True, field_type=int),
    "loop_condition_variables_count": FieldSpec(required=True, field_type=int),
    "has_break": FieldSpec(required=True, field_type=bool),
    "loop_condition_always_true": FieldSpec(required=True, field_type=bool),
    "initial_sat_condition": FieldSpec(required=True, field_type=bool),
    "array_operator": FieldSpec(required=True, field_type=bool),
    "pointer_operator": FieldSpec(required=True, field_type=bool),
    "summary": FieldSpec(required=True, field_type=str),
}

# Loop entry schemas (for nested validation)
LOOP_ENTRY_SCHEMA = {
    "id": FieldSpec(required=True, field_type=int, min_value=1),
    "code": FieldSpec(required=True, field_type=str),
}

INVARIANT_ENTRY_SCHEMA = {
    "loop_id": FieldSpec(required=True, field_type=int, min_value=1),
    "code": FieldSpec(required=True, field_type=str),
    "invariants": FieldSpec(required=True, field_type=list),
}

RANKING_ENTRY_SCHEMA = {
    "loop_id": FieldSpec(required=True, field_type=int, min_value=1),
    "code": FieldSpec(required=True, field_type=str),
    "invariants": FieldSpec(required=True, field_type=list),
    "explanation": FieldSpec(required=True, field_type=str),
    # Optional fields based on mode
    "ranking_function": FieldSpec(required=False, field_type=str),
    "template_type": FieldSpec(required=False, field_type=str),
    "template_depth": FieldSpec(required=False, field_type=int, min_value=1),
    "status": FieldSpec(required=False, field_type=str),
}


# ============================================================================
# Validation Result
# ============================================================================

@dataclass
class ValidationResult:
    """Result of YAML validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    yaml_type: Optional[str] = None
    
    def __bool__(self) -> bool:
        return self.valid
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        if self.yaml_type:
            lines.append(f"YAML Type: {self.yaml_type}")
        lines.append(f"Valid: {self.valid}")
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  - {err}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  - {warn}")
        return "\n".join(lines)


# ============================================================================
# YAML Type Detection
# ============================================================================

_LOOP_REF_PATTERN = re.compile(r"LOOP\s*(?:\{|\()?\s*(\d+)\s*(?:\}|\))?", re.IGNORECASE)

_YAML_TYPE_PATTERNS = {
    "extract": [r"_extract\.ya?ml$", r"_ext\.ya?ml$"],
    "invariant": [r"_inv\.ya?ml$", r"_invariant\.ya?ml$"],
    "ranking": [r"_ranking\.ya?ml$"],
    "feature": [r"_feature\.ya?ml$"],
}


def detect_yaml_type(path: Path, content: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Detect YAML type from filename or content.
    
    Args:
        path: File path
        content: Parsed YAML content (optional, for ambiguous cases)
    
    Returns:
        YAML type string or None if unknown
    """
    # Try filename pattern matching first
    name_lower = path.name.lower()
    for yaml_type, patterns in _YAML_TYPE_PATTERNS.items():
        if any(re.search(pattern, name_lower) for pattern in patterns):
            return yaml_type
    
    # Fallback to content-based detection
    if content and isinstance(content, dict):
        task = content.get("task", "").lower()
        if "extract" in task:
            return "extract"
        elif "invariant" in task:
            return "invariant"
        elif "ranking" in task:
            return "ranking"
        elif "feature" in task:
            return "feature"
        
        # Check for distinctive keys
        if "loops" in content and "loops_ids" in content:
            return "extract"
        elif "invariants_result" in content:
            return "invariant"
        elif "ranking_results" in content:
            return "ranking"
    
    return None


# ============================================================================
# Schema Validation
# ============================================================================

def validate_field(
    field_name: str,
    value: Any,
    spec: FieldSpec,
    path_context: str = ""
) -> List[str]:
    """Validate a single field against its specification.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    context = f"{path_context}.{field_name}" if path_context else field_name
    
    # Type validation
    if spec.field_type and value is not None:
        if not isinstance(value, spec.field_type):
            errors.append(
                f"{context}: expected type {spec.field_type.__name__}, "
                f"got {type(value).__name__}"
            )
    
    # Range validation for integers
    if isinstance(value, int):
        if spec.min_value is not None and value < spec.min_value:
            errors.append(f"{context}: value {value} < minimum {spec.min_value}")
        if spec.max_value is not None and value > spec.max_value:
            errors.append(f"{context}: value {value} > maximum {spec.max_value}")
    
    # Allowed values validation
    if spec.allowed_values and value not in spec.allowed_values:
        errors.append(
            f"{context}: value '{value}' not in allowed values: "
            f"{spec.allowed_values}"
        )
    
    return errors


def validate_schema(
    content: Dict[str, Any],
    schema: Dict[str, FieldSpec],
    path_context: str = ""
) -> List[str]:
    """Validate content against schema.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check required fields
    for field_name, spec in schema.items():
        if spec.required and field_name not in content:
            errors.append(f"{path_context}: missing required field '{field_name}'")
        elif field_name in content:
            errors.extend(validate_field(field_name, content[field_name], spec, path_context))
    
    return errors


# ============================================================================
# Loop ID Validation
# ============================================================================

def validate_loop_ids(entries: List[Dict[str, Any]], yaml_type: str) -> Tuple[List[str], List[str]]:
    """Validate loop ID consistency and nesting rules.
    
    Args:
        entries: List of loop/invariant/ranking entries
        yaml_type: Type of YAML (extract/invariant/ranking)
    
    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []
    
    if not entries:
        return errors, warnings
    
    # Collect loop IDs and their codes
    loop_data: List[Tuple[int, str]] = []
    for idx, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(f"Entry {idx} is not a dictionary")
            continue
        
        # Get loop ID
        if yaml_type == "extract":
            loop_id = entry.get("id")
        else:
            loop_id = entry.get("loop_id") or entry.get("id")
        
        if loop_id is None:
            errors.append(f"Entry {idx} missing loop ID")
            continue
        
        try:
            loop_id = int(loop_id)
        except (TypeError, ValueError):
            errors.append(f"Entry {idx} has invalid loop ID: {loop_id}")
            continue
        
        code = entry.get("code", "")
        loop_data.append((loop_id, str(code)))
    
    if not loop_data:
        return errors, warnings
    
    # Rule 1: Loop IDs should be sequential 1..N
    loop_ids = [lid for lid, _ in loop_data]
    expected_ids = list(range(1, len(loop_ids) + 1))
    
    if loop_ids != expected_ids:
        errors.append(
            f"Loop IDs are not sequential 1..N. "
            f"Expected {expected_ids}, got {loop_ids}"
        )
    
    # Rule 2: Check for duplicate IDs
    seen_ids = set()
    for loop_id, _ in loop_data:
        if loop_id in seen_ids:
            errors.append(f"Duplicate loop ID: {loop_id}")
        seen_ids.add(loop_id)
    
    # Rule 3: Nesting rule - LOOP{n} can only reference loop IDs that appeared before
    seen_loop_ids: Set[int] = set()
    for loop_id, code in loop_data:
        # Extract LOOP{n} references
        referenced_ids = set()
        for match in _LOOP_REF_PATTERN.findall(code):
            try:
                ref_id = int(match)
                referenced_ids.add(ref_id)
            except (TypeError, ValueError):
                continue
        
        # Check forward references
        for ref_id in referenced_ids:
            if ref_id not in seen_loop_ids:
                errors.append(
                    f"Loop {loop_id} references LOOP{ref_id} before it appears "
                    f"(seen so far: {sorted(seen_loop_ids)})"
                )
            elif ref_id == loop_id:
                warnings.append(f"Loop {loop_id} contains self-reference LOOP{ref_id}")
        
        seen_loop_ids.add(loop_id)
    
    return errors, warnings


# ============================================================================
# Main Validation Functions
# ============================================================================

def validate_yaml_file(path: Path, strict: bool = True) -> ValidationResult:
    """Validate a YAML file against its schema.
    
    Args:
        path: Path to YAML file
        strict: If True, treat warnings as errors
    
    Returns:
        ValidationResult object
    """
    errors = []
    warnings = []
    
    # Load YAML
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
    except Exception as e:
        return ValidationResult(
            valid=False,
            errors=[f"Failed to parse YAML: {e}"],
            warnings=[]
        )
    
    if not isinstance(content, dict):
        return ValidationResult(
            valid=False,
            errors=["YAML root must be a dictionary"],
            warnings=[]
        )
    
    # Detect YAML type
    yaml_type = detect_yaml_type(path, content)
    if not yaml_type:
        warnings.append("Could not detect YAML type from filename or content")
        return ValidationResult(valid=True, errors=[], warnings=warnings)
    
    # Select schema
    schema_map = {
        "extract": EXTRACT_SCHEMA,
        "invariant": INVARIANT_SCHEMA,
        "ranking": RANKING_SCHEMA,
        "feature": FEATURE_SCHEMA,
    }
    schema = schema_map.get(yaml_type)
    if not schema:
        return ValidationResult(
            valid=True,
            errors=[],
            warnings=[f"No schema defined for type '{yaml_type}'"],
            yaml_type=yaml_type
        )
    
    # Validate top-level schema
    errors.extend(validate_schema(content, schema))
    
    # Validate loop entries if applicable
    if yaml_type in {"extract", "invariant", "ranking"}:
        entry_key = {
            "extract": "loops",
            "invariant": "invariants_result",
            "ranking": "ranking_results",
        }[yaml_type]
        
        entries = content.get(entry_key)
        if isinstance(entries, list) and entries:
            # Validate entry schema
            entry_schema = {
                "extract": LOOP_ENTRY_SCHEMA,
                "invariant": INVARIANT_ENTRY_SCHEMA,
                "ranking": RANKING_ENTRY_SCHEMA,
            }[yaml_type]
            
            for idx, entry in enumerate(entries, start=1):
                if isinstance(entry, dict):
                    entry_errors = validate_schema(
                        entry, 
                        entry_schema,
                        path_context=f"{entry_key}[{idx}]"
                    )
                    errors.extend(entry_errors)
            
            # Validate loop IDs
            id_errors, id_warnings = validate_loop_ids(entries, yaml_type)
            errors.extend(id_errors)
            warnings.extend(id_warnings)
    
    # Determine validity
    valid = len(errors) == 0
    if strict:
        valid = valid and len(warnings) == 0
    
    return ValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        yaml_type=yaml_type
    )


def validate_yaml_content(
    content: Dict[str, Any],
    yaml_type: str,
    strict: bool = True
) -> ValidationResult:
    """Validate YAML content (already parsed) against schema.
    
    Args:
        content: Parsed YAML content
        yaml_type: Type of YAML (extract/invariant/ranking/feature)
        strict: If True, treat warnings as errors
    
    Returns:
        ValidationResult object
    """
    errors = []
    warnings = []
    
    if not isinstance(content, dict):
        return ValidationResult(
            valid=False,
            errors=["YAML content must be a dictionary"],
            warnings=[]
        )
    
    # Select schema
    schema_map = {
        "extract": EXTRACT_SCHEMA,
        "invariant": INVARIANT_SCHEMA,
        "ranking": RANKING_SCHEMA,
        "feature": FEATURE_SCHEMA,
    }
    schema = schema_map.get(yaml_type)
    if not schema:
        return ValidationResult(
            valid=False,
            errors=[f"Unknown YAML type: {yaml_type}"],
            warnings=[]
        )
    
    # Validate schema
    errors.extend(validate_schema(content, schema))
    
    # Validate loop entries if applicable
    if yaml_type in {"extract", "invariant", "ranking"}:
        entry_key = {
            "extract": "loops",
            "invariant": "invariants_result",
            "ranking": "ranking_results",
        }[yaml_type]
        
        entries = content.get(entry_key)
        if isinstance(entries, list) and entries:
            entry_schema = {
                "extract": LOOP_ENTRY_SCHEMA,
                "invariant": INVARIANT_ENTRY_SCHEMA,
                "ranking": RANKING_ENTRY_SCHEMA,
            }[yaml_type]
            
            for idx, entry in enumerate(entries, start=1):
                if isinstance(entry, dict):
                    entry_errors = validate_schema(
                        entry,
                        entry_schema,
                        path_context=f"{entry_key}[{idx}]"
                    )
                    errors.extend(entry_errors)
            
            # Validate loop IDs
            id_errors, id_warnings = validate_loop_ids(entries, yaml_type)
            errors.extend(id_errors)
            warnings.extend(id_warnings)
    
    valid = len(errors) == 0
    if strict:
        valid = valid and len(warnings) == 0
    
    return ValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        yaml_type=yaml_type
    )


# ============================================================================
# Backward Compatibility Helpers
# ============================================================================

def get_missing_required_keys(path: Path, content: Dict[str, Any]) -> List[str]:
    """Legacy function: get list of missing required keys.
    
    This maintains compatibility with existing code using validate_yaml_required_keys.
    """
    yaml_type = detect_yaml_type(path, content)
    if not yaml_type:
        return []
    
    schema_map = {
        "extract": EXTRACT_SCHEMA,
        "invariant": INVARIANT_SCHEMA,
        "ranking": RANKING_SCHEMA,
        "feature": FEATURE_SCHEMA,
    }
    schema = schema_map.get(yaml_type, {})
    
    missing = []
    for field_name, spec in schema.items():
        if spec.required and field_name not in content:
            missing.append(field_name)
    
    return missing
