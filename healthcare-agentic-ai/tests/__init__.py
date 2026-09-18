"""Allow unittest discovery from both the workspace and project directories."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
