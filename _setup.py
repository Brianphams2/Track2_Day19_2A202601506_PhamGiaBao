"""Khởi tạo đường dẫn khi chạy các notebook bản nộp đặt tại repository root."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BIN_DIR = Path(sys.executable).parent
if str(BIN_DIR) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{BIN_DIR}{os.pathsep}{os.environ.get('PATH', '')}"

# Các notebook nguồn suy ra repository bằng `parent.parent` từ helper nằm
# trong thư mục `notebooks/`. Giữ cùng giao ước cho bản notebook đặt ở root.
__file__ = str(REPO_ROOT / "notebooks" / "_setup.py")
