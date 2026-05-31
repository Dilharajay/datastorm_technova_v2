from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_OUTLET_TYPE_MAPPING = {
    "grocry": "grocery",
    "grocery": "grocery",
    "bakry": "bakery",
    "bakery": "bakery",
    "hotel": "hotel",
    "pharmacy": "pharmacy",
    "kiosk": "kiosk",
    "eatery": "eatery",
    "smmt": "smmt",
}