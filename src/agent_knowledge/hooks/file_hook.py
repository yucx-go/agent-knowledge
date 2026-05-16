"""FileHook: monitor directory for file changes via os.stat polling."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Hook, HookResult

if TYPE_CHECKING:
    from ..core.compiler import Compiler
    from ..core.umsf import UMSFEvent
    from ..core.vault import Vault


class FileHook(Hook):
    """Fires on UMSF ``file_change`` events. Reads:

    - ``event.path`` — filesystem path of the changed file (required)

    Pairs with :class:`FileWatcher`, which polls a directory and fires
    ``file_change`` events for any modified file.
    """

    name = "file_change"
    event_types = ["file_change"]

    # Max file size to ingest (1 MB)
    MAX_FILE_SIZE = 1_048_576

    def should_fire(self, event: "UMSFEvent") -> bool:
        if event.type not in self.event_types:
            return False
        return bool(event.path) and os.path.isfile(event.path)

    def process(
        self,
        event: "UMSFEvent",
        session_id: str,
        vault: "Vault",
        compiler: "Compiler",
    ) -> HookResult:
        path = event.path
        p = Path(path)

        # Safety: skip large or binary files
        if p.stat().st_size > self.MAX_FILE_SIZE:
            return HookResult()

        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return HookResult()

        source = compiler.ingest(
            text,
            title=p.name,
            source_type="file",
            path=str(p),
        )

        return HookResult(
            claims=[],
            source_id=source.id,
            entities_found=source.entities_extracted,
        )


class FileWatcher:
    """Polls a directory for file changes using os.stat.

    Zero external dependencies — no watchdog/inotify. Detected changes
    are wrapped as UMSF ``file_change`` events and dispatched through
    :class:`HookManager`.
    """

    def __init__(
        self,
        watch_dir: "str | Path",
        manager: "HookManager",  # type: ignore[name-defined]
        interval: float = 5.0,
        patterns: tuple[str, ...] = ("*.md", "*.txt", "*.yaml", "*.yml"),
    ):
        self.watch_dir = Path(watch_dir).expanduser().resolve()
        self.manager = manager
        self.interval = interval
        self.patterns = patterns
        self._mtimes: dict[str, float] = {}
        self._running = False
        self._initialized = False  # True after first full scan

    def _scan(self) -> list[str]:
        """Return list of changed file paths since last scan."""
        changed: list[str] = []
        current_files: set[str] = set()

        for pattern in self.patterns:
            for p in self.watch_dir.rglob(pattern):
                if not p.is_file():
                    continue
                key = str(p)
                current_files.add(key)
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if key not in self._mtimes or self._mtimes[key] < mtime:
                    if self._initialized:  # skip first full scan
                        changed.append(key)
                    self._mtimes[key] = mtime

        # Remove deleted files from tracking
        for key in list(self._mtimes):
            if key not in current_files:
                del self._mtimes[key]

        return changed

    def poll_once(self) -> list[str]:
        """Run a single poll cycle. Returns list of changed paths that were fired."""
        from ..core.umsf import UMSFDocument, UMSFEvent

        changed = self._scan()
        if not self._initialized:
            self._initialized = True
        for path in changed:
            ev = UMSFEvent(type="file_change", ts=time.time(), path=path)
            doc = UMSFDocument(
                agent="file-watcher",
                source_type="file_change",
                events=[ev],
            )
            self.manager.fire(doc)
        return changed

    def run(self) -> None:
        """Blocking polling loop. Call ``stop()`` from another thread to exit."""
        self._running = True
        # Initial scan to populate mtimes (no events fired)
        self._scan()
        while self._running:
            self.poll_once()
            time.sleep(self.interval)

    def stop(self) -> None:
        self._running = False
