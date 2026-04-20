"""Unified belief scoring with temporal decay and conflict detection."""

import time
from dataclasses import dataclass, field
from enum import Enum

class BeliefState(Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"

@dataclass
class BeliefRecord:
    topic: str
    confidence: float
    trust: float
    relevance: float
    source: str = ""
    timestamp: float = field(default_factory=time.time)

class UnifiedBelief:
    def __init__(self, decay_rate: float = 0.99):
        self.decay_rate = decay_rate
        self._beliefs: dict[str, list[BeliefRecord]] = {}

    def record(self, topic: str, confidence: float, trust: float = 0.5,
               relevance: float = 0.5, source: str = "") -> BeliefRecord:
        rec = BeliefRecord(topic=topic, confidence=confidence, trust=trust,
                           relevance=relevance, source=source)
        if topic not in self._beliefs:
            self._beliefs[topic] = []
        self._beliefs[topic].append(rec)
        return rec

    def score(self, topic: str) -> float:
        records = self._beliefs.get(topic, [])
        if not records: return 0.0
        total, weight = 0.0, 0.0
        for r in records:
            w = self._time_weight(r.timestamp)
            total += (r.confidence * r.trust * r.relevance) * w
            weight += w
        return total / max(weight, 1e-9)

    def state(self, topic: str) -> BeliefState:
        records = self._beliefs.get(topic, [])
        if not records: return BeliefState.UNKNOWN
        s = self.score(topic)
        if s >= 0.7: return BeliefState.STRONG
        if s >= 0.4: return BeliefState.MODERATE
        if s >= 0.2: return BeliefState.WEAK
        return BeliefState.UNKNOWN

    def detect_conflict(self, topic: str, threshold: float = 0.3) -> bool:
        records = self._beliefs.get(topic, [])
        if len(records) < 2: return False
        scores = [r.confidence * r.trust for r in records[-10:]]
        if not scores: return False
        return (max(scores) - min(scores)) > threshold

    def decay(self):
        now = time.time()
        for topic in self._beliefs:
            cutoff = now - 86400 * 7
            self._beliefs[topic] = [r for r in self._beliefs[topic] if r.timestamp >= cutoff]

    def top_topics(self, n: int = 10) -> list[tuple[str, float, BeliefState]]:
        scored = [(t, self.score(t), self.state(t)) for t in self._beliefs]
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def _time_weight(self, timestamp: float) -> float:
        age = time.time() - timestamp
        return self.decay_rate ** (age / 3600)

    @property
    def stats(self) -> dict:
        states = {}
        for t in self._beliefs:
            s = self.state(t)
            states[s.value] = states.get(s.value, 0) + 1
        return {"topics": len(self._beliefs), "states": states,
                "total_records": sum(len(v) for v in self._beliefs.values())}
