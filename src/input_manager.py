"""
src/input_manager.py
Classifies, validates, and organises uploaded files.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config.settings import (
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_VIDEO_EXTS,
    UPLOAD_DIR,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UploadedFile:
    """Metadata about a single uploaded file."""
    original_name: str
    saved_path: Path
    file_type: str          # "image" | "video"
    size_bytes: int
    sha256: str
    # Populated later by processors
    width: Optional[int] = None
    height: Optional[int] = None
    duration_s: Optional[float] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    error: Optional[str] = None

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    @property
    def resolution(self) -> Optional[str]:
        if self.width and self.height:
            return f"{self.width} × {self.height}"
        return None


@dataclass
class InputSession:
    """Holds all files for one processing session."""
    files: List[UploadedFile] = field(default_factory=list)

    @property
    def images(self) -> List[UploadedFile]:
        return [f for f in self.files if f.file_type == "image"]

    @property
    def videos(self) -> List[UploadedFile]:
        return [f for f in self.files if f.file_type == "video"]

    @property
    def total_count(self) -> int:
        return len(self.files)


# ---------------------------------------------------------------------------
# File classification helpers
# ---------------------------------------------------------------------------

def classify_extension(filename: str) -> Optional[str]:
    """Return 'image', 'video', or None for unsupported extension."""
    ext = Path(filename).suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTS:
        return "image"
    if ext in SUPPORTED_VIDEO_EXTS:
        return "video"
    return None


def _sha256(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main manager
# ---------------------------------------------------------------------------

class InputManager:
    """Accepts Streamlit UploadedFile objects, saves them, builds metadata."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = UPLOAD_DIR / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def process_uploads(self, st_files: list) -> InputSession:
        """
        Accept a list of Streamlit UploadedFile objects.
        Save to disk, classify, compute hashes.
        """
        session = InputSession()
        for st_file in st_files:
            result = self._handle_file(st_file)
            if result:
                session.files.append(result)
        return session

    def _handle_file(self, st_file) -> Optional[UploadedFile]:
        file_type = classify_extension(st_file.name)
        if file_type is None:
            return UploadedFile(
                original_name=st_file.name,
                saved_path=Path("/dev/null"),
                file_type="unsupported",
                size_bytes=0,
                sha256="",
                error=f"Unsupported file type: {Path(st_file.name).suffix}",
            )

        dest = self.session_dir / st_file.name
        # Avoid collision
        if dest.exists():
            stem = dest.stem
            suf = dest.suffix
            dest = self.session_dir / f"{stem}_{hashlib.md5(st_file.name.encode()).hexdigest()[:6]}{suf}"

        data = st_file.read()
        dest.write_bytes(data)

        return UploadedFile(
            original_name=st_file.name,
            saved_path=dest,
            file_type=file_type,
            size_bytes=len(data),
            sha256=_sha256(dest),
        )

    def clear_session(self) -> None:
        """Remove session upload directory."""
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir)
