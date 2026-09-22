"""Terminal entry point; no changes to the Phase 5/6 workflow implementation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.terminal import main


if __name__ == '__main__':
    raise SystemExit(main())
