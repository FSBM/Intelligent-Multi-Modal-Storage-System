"""
JSON Analyzer
Finds structure, patterns, and repeated fields in JSON data
Handles batch JSON by merging structures
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class JSONAnalyzer:
    """Analyzes JSON structure to determine patterns and consistency"""
    
    def __init__(self):
        self.min_consistency_threshold = 0.7  # 70% field consistency for SQL
        self.max_nesting_depth = 5
    
    async def analyze(self, file_path: Path) -> Dict:
        """
        Analyze JSON file and return structure analysis
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle single object vs array
            if isinstance(data, list):
                return await self._analyze_batch(data)
            else:
                return await self._analyze_single(data)
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            raise ValueError(f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error analyzing JSON {file_path}: {e}", exc_info=True)
            raise
    
    async def _analyze_single(self, obj: Dict) -> Dict:
        """Analyze a single JSON object"""
        structure = self._extract_structure(obj)
        patterns = self._detect_patterns([obj])
        
        return {
            "type": "single",
            "structure": structure,
            "patterns": patterns,
            "field_count": len(structure.get("fields", {})),
            "nesting_depth": structure.get("max_depth", 0),
            "is_consistent": True,  # Single object is always consistent
        }
    
    async def _analyze_batch(self, objects: List[Dict]) -> Dict:
        """Analyze a batch of JSON objects"""
        if not objects:
            raise ValueError("Empty JSON array")
        
        # Extract structure from all objects
        structures = [self._extract_structure(obj) for obj in objects]
        
        # Merge structures
        merged_structure = self._merge_structures(structures)
        
        # Detect patterns across batch
        patterns = self._detect_patterns(objects)
        
        # Calculate consistency
        consistency = self._calculate_consistency(objects, merged_structure)
        
        return {
            "type": "batch",
            "count": len(objects),
            "structure": merged_structure,
            "patterns": patterns,
            "consistency": consistency,
            "field_count": len(merged_structure.get("fields", {})),
            "nesting_depth": merged_structure.get("max_depth", 0),
            "is_consistent": consistency["overall"] >= self.min_consistency_threshold,
        }
    
    def _extract_structure(self, obj: Any, depth: int = 0, path: str = "") -> Dict:
        """Recursively extract structure from JSON object"""
        if depth > self.max_nesting_depth:
            return {"type": "truncated", "path": path}
        
        if isinstance(obj, dict):
            fields = {}
            max_depth = depth
            
            for key, value in obj.items():
                field_path = f"{path}.{key}" if path else key
                field_info = self._extract_structure(value, depth + 1, field_path)
                fields[key] = field_info
                if isinstance(field_info, dict):
                    max_depth = max(max_depth, field_info.get("max_depth", depth))
            
            return {
                "type": "object",
                "fields": fields,
                "max_depth": max_depth,
            }
        
        elif isinstance(obj, list):
            if not obj:
                return {"type": "array", "item_type": "unknown", "empty": True}
            
            # Analyze first few items to determine structure
            item_structures = [
                self._extract_structure(item, depth + 1, path)
                for item in obj[:10]  # Sample first 10 items
            ]
            
            # Determine if items are consistent
            item_types = [s.get("type") if isinstance(s, dict) else type(s).__name__ 
                         for s in item_structures]
            most_common_type = Counter(item_types).most_common(1)[0][0]
            
            return {
                "type": "array",
                "item_type": most_common_type,
                "sample_structure": item_structures[0] if item_structures else None,
                "max_depth": max(
                    (s.get("max_depth", depth) for s in item_structures if isinstance(s, dict)),
                    default=depth
                ),
            }
        
        else:
            # Primitive type
            return {
                "type": type(obj).__name__,
                "value_sample": str(obj)[:50] if obj is not None else None,
            }
    
    def _merge_structures(self, structures: List[Dict]) -> Dict:
        """Merge multiple structures into one"""
        if not structures:
            return {"type": "empty"}
        
        if len(structures) == 1:
            return structures[0]
        
        # For objects, merge fields
        if all(s.get("type") == "object" for s in structures):
            all_fields = set()
            for s in structures:
                all_fields.update(s.get("fields", {}).keys())
            
            merged_fields = {}
            for field in all_fields:
                field_structures = [
                    s.get("fields", {}).get(field)
                    for s in structures
                    if field in s.get("fields", {})
                ]
                
                if field_structures:
                    # Merge field structures
                    merged_fields[field] = self._merge_structures(field_structures)
            
            max_depth = max(
                (s.get("max_depth", 0) for s in structures),
                default=0
            )
            
            return {
                "type": "object",
                "fields": merged_fields,
                "max_depth": max_depth,
            }
        
        # For other types, return the first structure
        return structures[0]
    
    def _detect_patterns(self, objects: List[Dict]) -> Dict:
        """Detect patterns in JSON objects"""
        patterns = {
            "repeated_fields": [],
            "optional_fields": [],
            "value_patterns": {},
            "array_lengths": [],
        }
        
        if not objects:
            return patterns
        
        # Collect all field names
        all_fields = set()
        field_presence = defaultdict(int)
        
        for obj in objects:
            fields = self._get_all_fields(obj)
            all_fields.update(fields)
            for field in fields:
                field_presence[field] += 1
        
        # Determine required vs optional fields
        total = len(objects)
        for field in all_fields:
            presence_ratio = field_presence[field] / total
            if presence_ratio >= 0.9:  # Present in 90%+ of objects
                patterns["repeated_fields"].append(field)
            elif presence_ratio < 0.5:
                patterns["optional_fields"].append(field)
        
        # Detect value patterns (e.g., email, URL, date)
        for field in patterns["repeated_fields"][:10]:  # Check top 10 fields
            values = [
                self._get_nested_value(obj, field)
                for obj in objects
                if self._has_field(obj, field)
            ]
            pattern_type = self._detect_value_pattern(values)
            if pattern_type:
                patterns["value_patterns"][field] = pattern_type
        
        return patterns
    
    def _get_all_fields(self, obj: Any, prefix: str = "") -> Set[str]:
        """Get all field paths from an object"""
        fields = set()
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                field_path = f"{prefix}.{key}" if prefix else key
                fields.add(field_path)
                fields.update(self._get_all_fields(value, field_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:5]):  # Sample first 5 items
                fields.update(self._get_all_fields(item, f"{prefix}[{i}]"))
        
        return fields
    
    def _has_field(self, obj: Any, field_path: str) -> bool:
        """Check if object has a field at the given path"""
        parts = field_path.split(".")
        current = obj
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        return True
    
    def _get_nested_value(self, obj: Any, field_path: str) -> Any:
        """Get value at nested field path"""
        parts = field_path.split(".")
        current = obj
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    
    def _detect_value_pattern(self, values: List[Any]) -> Optional[str]:
        """Detect pattern in values (email, URL, date, etc.)"""
        if not values:
            return None
        
        # Check for email pattern
        import re
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if all(isinstance(v, str) and email_pattern.match(v) for v in values[:5]):
            return "email"
        
        # Check for URL pattern
        url_pattern = re.compile(r'^https?://')
        if all(isinstance(v, str) and url_pattern.match(v) for v in values[:5]):
            return "url"
        
        # Check for date pattern
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}')
        if all(isinstance(v, str) and date_pattern.match(v) for v in values[:5]):
            return "date"
        
        return None
    
    def _calculate_consistency(self, objects: List[Dict], structure: Dict) -> Dict:
        """Calculate consistency metrics"""
        if not objects:
            return {"overall": 0.0}
        
        # Get all fields from structure
        structure_fields = set(self._get_all_fields_from_structure(structure))
        
        # Calculate field presence across objects
        field_presence = defaultdict(int)
        for obj in objects:
            obj_fields = self._get_all_fields(obj)
            for field in structure_fields:
                if field in obj_fields:
                    field_presence[field] += 1
        
        # Calculate consistency ratio
        total = len(objects)
        consistency_scores = [
            field_presence[field] / total
            for field in structure_fields
        ]
        
        overall_consistency = (
            sum(consistency_scores) / len(consistency_scores)
            if consistency_scores else 0.0
        )
        
        return {
            "overall": overall_consistency,
            "field_consistency": dict(field_presence),
            "total_fields": len(structure_fields),
        }
    
    def _get_all_fields_from_structure(self, structure: Dict, prefix: str = "") -> List[str]:
        """Extract all field paths from structure"""
        fields = []
        
        if structure.get("type") == "object":
            for field_name, field_structure in structure.get("fields", {}).items():
                field_path = f"{prefix}.{field_name}" if prefix else field_name
                fields.append(field_path)
                fields.extend(self._get_all_fields_from_structure(field_structure, field_path))
        
        return fields

