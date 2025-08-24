# progress_utils.py
import sys
import time
from contextlib import contextmanager

class ProgressTask:
    def __init__(self, enabled: bool, label: str, total: int | None, stream=None):
        self.enabled = enabled
        self.label = label
        self.total = total
        self.n = 0
        self.stream = stream or sys.stderr
        self._last_render = 0.0
        if self.enabled:
            self._render(force=True)

    def update(self, inc: int = 1):
        if not self.enabled:
            return
        self.n += inc
        # rate-limit rendering to avoid spam
        now = time.time()
        if now - self._last_render >= 0.05:  # 20 FPS max
            self._render()
            self._last_render = now

    def close(self):
        if not self.enabled:
            return
        self._render(done=True)
        self.stream.write("\n")
        self.stream.flush()

    def _render(self, force: bool = False, done: bool = False):
        if not self.enabled:
            return
        if self.total:
            pct = min(100, int(100 * self.n / max(1, self.total)))
            bar_len = 24
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "·" * (bar_len - filled)
            msg = f"\r[{bar}] {pct:3d}% {self.label}"
        else:
            dots = "." * (self.n % 10)
            msg = f"\r{self.label} {dots}"
        if done and self.total:
            msg = msg.replace("]", "]✓", 1)
        self.stream.write(msg)
        self.stream.flush()

    # context manager sugar
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()

class ProgressReporter:
    def __init__(self, enabled: bool = False, stream=None):
        self.enabled = enabled
        self.stream = stream or sys.stderr

    def start(self, label: str, total: int | None = None) -> ProgressTask:
        return ProgressTask(self.enabled, label, total, stream=self.stream)

    def mark(self, text: str):
        if not self.enabled:
            return
        self.stream.write(text + "\n")
        self.stream.flush()
