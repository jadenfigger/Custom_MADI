"""Bounded, thread-safe resources for the single-process local explorer."""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np


class ArrayCache:
    """LRU bounded by array bytes; returned arrays are immutable.

    Loading under the lock avoids duplicate multi-GB reads when callbacks
    request the same block concurrently. Oversized values are never retained.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes = 0
        self._items = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key, loader):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key][0]
            value = loader()
            arrays = value if isinstance(value, tuple) else (value,)
            size = sum(x.nbytes for x in arrays if isinstance(x, np.ndarray))
            for array in arrays:
                if isinstance(array, np.ndarray):
                    array.setflags(write=False)
            if 0 < self.max_bytes and size <= self.max_bytes:
                # The item cap also bounds metadata for empty Jacobian batches.
                while self._items and (self.bytes + size > self.max_bytes or len(self._items) >= 128):
                    _, (_, old_size) = self._items.popitem(last=False)
                    self.bytes -= old_size
                self._items[key] = (value, size)
                self.bytes += size
            return value


class Superseded(Exception):
    """A newer request replaced this computation."""


@dataclass
class Job:
    token: str
    touched: float
    cancel: threading.Event = field(default_factory=threading.Event)
    future: object = None


class Jobs:
    """One latest request per browser; bounded retention and cooperative cancel.

    Workers never mutate Dash state. Only polling the latest token publishes
    results. Queued obsolete futures are cancelled; running work checks its
    cancellation event between stages. Intended for the local threaded server,
    not a multi-process WSGI deployment.
    """

    def __init__(self, workers=2, max_sessions=8, ttl=1800):
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="manifold")
        self._lock = threading.RLock()
        self._jobs = {}
        self.max_sessions = max_sessions
        self.ttl = ttl

    def submit(self, session, function):
        with self._lock:
            now = time.monotonic()
            for key, job in list(self._jobs.items()):
                if now - job.touched > self.ttl:
                    job.cancel.set()
                    job.future.cancel()
                    del self._jobs[key]
            old = self._jobs.get(session)
            if old:
                old.cancel.set()
                old.future.cancel()
            elif len(self._jobs) >= self.max_sessions:
                completed = [(key, old) for key, old in self._jobs.items() if old.future.done()]
                if not completed:
                    raise ValueError("All exploration slots are computing. Try again when a calculation finishes.")
                key, _ = min(completed, key=lambda item: item[1].touched)
                del self._jobs[key]
            job = Job(uuid.uuid4().hex, now)

            def checkpoint():
                if job.cancel.is_set():
                    raise Superseded()

            def run():
                checkpoint()
                result = function(checkpoint)
                checkpoint()
                return result

            job.future = self._pool.submit(run)
            self._jobs[session] = job
            return job.token

    def poll(self, session, token):
        with self._lock:
            job = self._jobs.get(session)
            if job is None or job.token != token:
                return "expired", None
            job.touched = time.monotonic()
            if not job.future.done():
                return "running" if job.future.running() else "queued", None
            try:
                return "complete", job.future.result()
            except Superseded:
                return "expired", None
            except Exception as exc:
                return "error", str(exc)

    def invalidate(self, session):
        with self._lock:
            job = self._jobs.pop(session, None)
            if job:
                job.cancel.set()
                job.future.cancel()

    def close(self):
        with self._lock:
            for job in self._jobs.values():
                job.cancel.set()
            self._jobs.clear()
        self._pool.shutdown(wait=True, cancel_futures=True)
