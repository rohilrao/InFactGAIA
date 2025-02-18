# @title
import numpy as np
from dataclasses import dataclass
import math
from typing import List, Tuple, Dict, Optional
import json

@dataclass
class DataPoint:
    raw_data: str  # Original data representation
    metadata: dict  # Source information, timestamps, etc.
    likelihood_plus: float  # l^+_i
    likelihood_minus: float  # l^-_i
    posterior: float  # π_i
    analysis_rationale: str  # LLM explanation for likelihood calculation
