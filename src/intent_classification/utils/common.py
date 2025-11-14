"""
Common utilities
"""
import logging
import random
import numpy as np
from typing import Any, Dict, List
import json
from pathlib import Path
import math
import datetime as dt
import decimal
import enum
import pathlib
from dataclasses import is_dataclass, asdict
import numpy as np
import pandas as pd


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable Python primitives:
    - dict, list, tuple, set, frozenset -> converted recursively (sets become lists)
    - str, int, float, bool, None -> kept (NaN/Inf handled)
    - datetime/date/time/timedelta -> ISO strings
    - decimal.Decimal -> float (or str if not finite)
    - enum.Enum -> value
    - pathlib.Path -> str
    - dataclasses -> dict via asdict
    - NumPy scalars/arrays -> Python scalars / lists
    - pandas Series/DataFrame/Index -> dict/list (records-orient for DataFrame)

    Note: NaN and +/-Inf are converted to None by default (valid JSON).
    """

    # Fast path for basic JSON-native types
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    # Handle floats and special values
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return None  # replace NaN/Inf with null

    # Dataclasses
    if is_dataclass(obj):
        return sanitize_for_json(asdict(obj))

    # Enums
    if isinstance(obj, enum.Enum):
        return sanitize_for_json(obj.value)

    # Decimal
    if isinstance(obj, decimal.Decimal):
        try:
            f = float(obj)
            return f if math.isfinite(f) else None
        except Exception:
            return str(obj)

    # Dates and times
    if isinstance(obj, (dt.datetime, dt.date, dt.time)):
        # Use ISO 8601
        return obj.isoformat()
    if isinstance(obj, dt.timedelta):
        return obj.total_seconds()

    # Paths
    if isinstance(obj, pathlib.Path):
        return str(obj)

    # NumPy support (if available)
    if np is not None:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            f = float(obj)
            return f if math.isfinite(f) else None
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            # Convert to list then sanitize elements
            return sanitize_for_json(obj.tolist())
        # Handle numpy scalar datetime/timedelta
        if isinstance(obj, (np.datetime64,)):
            # Convert to Python datetime via pandas if present, else string
            try:
                if pd is not None:
                    return pd.to_datetime(obj).to_pydatetime().isoformat()
            except Exception:
                pass
            return str(obj)
        if isinstance(obj, (np.timedelta64,)):
            try:
                seconds = obj / np.timedelta64(1, "s")
                return float(seconds)
            except Exception:
                return str(obj)

    # pandas support (if available)
    if pd is not None:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Timedelta):
            return obj.total_seconds()
        if isinstance(obj, pd.Series):
            # Convert to dict to preserve index labels
            return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, pd.Index):
            return [sanitize_for_json(v) for v in obj.tolist()]
        if isinstance(obj, pd.DataFrame):
            # Records orientation is usually the most interoperable
            return [sanitize_for_json(r) for r in obj.to_dict(orient="records")]

    # Containers
    if isinstance(obj, dict):
        return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [sanitize_for_json(v) for v in obj]

    # Fallback: string representation
    return str(obj)

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def save_json(data: Any, file_path: str):
    """Save data as JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path: str) -> Any:
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(path: str):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)

def print_metrics_summary(metrics: Dict[str, Any]):
    """Print a formatted summary of metrics"""
    print("\n=== Classification Metrics ===")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Macro F1: {metrics.get('macro_f1', 0):.4f}")
    print(f"Weighted F1: {metrics.get('weighted_f1', 0):.4f}")
    
    if 'per_class' in metrics:
        print("\n--- Per-Class Metrics ---")
        for label, class_metrics in metrics['per_class'].items():
            print(f"{label:>12}: P={class_metrics['precision']:.3f} R={class_metrics['recall']:.3f} F1={class_metrics['f1']:.3f} Support={class_metrics['support']}")