from __future__ import annotations

import heapq


class Pacer:
    """Schedules one send per active camera every 1/fps seconds.

    New cameras are phase-spread across the interval so sends do not arrive in bursts, and a
    camera that falls behind skips missed slots instead of catching up in a burst.
    """

    def __init__(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.interval = 1.0 / fps
        self._active = 0
        self._heap: list[tuple[float, int]] = []
        self._scheduled: set[int] = set()

    @property
    def active(self) -> int:
        return self._active

    def set_active(self, count: int, now: float) -> None:
        count = max(0, count)
        new_indices = [index for index in range(count) if index not in self._scheduled]
        for position, index in enumerate(new_indices):
            phase = self.interval * position / max(1, len(new_indices))
            heapq.heappush(self._heap, (now + phase, index))
            self._scheduled.add(index)
        self._active = count
        # Pausing (count 0) clears the schedule, so resuming phase-spreads again instead of bursting.
        self._drop_inactive()

    def next_due(self) -> float | None:
        self._drop_inactive()
        return self._heap[0][0] if self._heap else None

    def due(self, now: float) -> list[int]:
        ready: list[int] = []
        while self._heap and self._heap[0][0] <= now:
            due_at, index = heapq.heappop(self._heap)
            if index >= self._active:
                self._scheduled.discard(index)
                continue
            ready.append(index)
            next_at = due_at + self.interval
            if next_at <= now:
                next_at = now + self.interval
            heapq.heappush(self._heap, (next_at, index))
        return ready

    def _drop_inactive(self) -> None:
        while self._heap and self._heap[0][1] >= self._active:
            _, index = heapq.heappop(self._heap)
            self._scheduled.discard(index)
