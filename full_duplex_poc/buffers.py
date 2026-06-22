from __future__ import annotations

from collections import deque

import numpy as np


class RollingAudioBuffer:
    def __init__(self, max_samples: int) -> None:
        self.max_samples = int(max_samples)
        self._chunks: deque[np.ndarray] = deque()
        self._size = 0

    def append(self, audio: np.ndarray) -> None:
        if audio is None:
            return

        array = np.asarray(audio, dtype=np.float32).ravel()
        if array.size == 0:
            return

        self._chunks.append(array.copy())
        self._size += int(array.size)
        self._trim_left()

    def extend(self, chunks: list[np.ndarray]) -> None:
        for chunk in chunks:
            self.append(chunk)

    def _trim_left(self) -> None:
        while self._size > self.max_samples and self._chunks:
            left = self._chunks[0]
            overflow = self._size - self.max_samples
            if overflow >= len(left):
                self._chunks.popleft()
                self._size -= len(left)
                continue

            self._chunks[0] = left[overflow:]
            self._size -= overflow
            break

    def clear(self) -> None:
        self._chunks.clear()
        self._size = 0

    def to_numpy(self) -> np.ndarray:
        if not self._chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self._chunks)).astype(np.float32, copy=False)

    def tail(self, num_samples: int) -> np.ndarray:
        array = self.to_numpy()
        if num_samples <= 0:
            return np.array([], dtype=np.float32)
        return array[-num_samples:]

    def consume(self, num_samples: int) -> np.ndarray:
        if num_samples <= 0 or self._size == 0:
            return np.array([], dtype=np.float32)

        num_samples = min(num_samples, self._size)
        collected: list[np.ndarray] = []
        remaining = num_samples

        while remaining > 0 and self._chunks:
            left = self._chunks[0]
            if len(left) <= remaining:
                collected.append(left)
                self._chunks.popleft()
                self._size -= len(left)
                remaining -= len(left)
                continue

            collected.append(left[:remaining])
            self._chunks[0] = left[remaining:]
            self._size -= remaining
            remaining = 0

        if not collected:
            return np.array([], dtype=np.float32)
        return np.concatenate(collected).astype(np.float32, copy=False)

    @property
    def size(self) -> int:
        return self._size

