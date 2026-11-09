import re
from typing import List


def should_exclude(key: str, regexes: List[str]) -> bool:
    """Returns True if the key matches any of the exclusion patterns."""
    return any(re.compile(pattern).search(key) for pattern in regexes)
