"""Chapter naming helpers shared by builders and reports."""

import re
from pathlib import Path


def chapter_number(path):
    """Return an arbitrary-length leading chapter number, or ``None``."""
    match = re.match(r"^(\d+)(?:[_\-\s]|$)", Path(path).stem)
    return int(match.group(1)) if match else None
