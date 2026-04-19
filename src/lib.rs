//! plato-unified-belief — Unified belief system
//!
//! Three separate Bayesian systems for three different purposes, unified:
//! - Confidence (sensor fusion, from flux-confidence)
//! - Trust (agent reliability, from cuda-trust)
//! - Tile weight (knowledge relevance, from plato-tiling)
//!
//! All three use the same math: Bayesian update with temporal decay.
//! This crate unifies them into one BeliefScore with three dimensions.

// ── Belief Dimensions ───────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeliefDimension {
    /// How confident are we in this information? (sensor fusion)
    Confidence,
    /// How reliable is the agent that produced this? (agent trust)
    Trust,
    /// How relevant is this knowledge to the current context? (tile weight)
    Relevance,
}

impl BeliefDimension {
    pub fn all() -> &'static [BeliefDimension] {
        &[BeliefDimension::Confidence, BeliefDimension::Trust, BeliefDimension::Relevance]
    }

    pub fn name(&self) -> &'static str {
        match self {
            BeliefDimension::Confidence => "confidence",
            BeliefDimension::Trust => "trust",
            BeliefDimension::Relevance => "relevance",
        }
    }
}

// ── Belief Score ─────────────────────────────────────────

/// A unified belief score with three dimensions.
/// Each dimension is 0.0-1.0 with Bayesian update + temporal decay.
#[derive(Debug, Clone, Copy)]
pub struct BeliefScore {
    pub confidence: f32,
    pub trust: f32,
    pub relevance: f32,
}

impl Default for BeliefScore {
    fn default() -> Self {
        Self { confidence: 0.5, trust: 0.5, relevance: 0.5 }
    }
}

impl BeliefScore {
    pub fn new(confidence: f32, trust: f32, relevance: f32) -> Self {
        Self {
            confidence: confidence.max(0.0).min(1.0),
            trust: trust.max(0.0).min(1.0),
            relevance: relevance.max(0.0).min(1.0),
        }
    }

    /// Get a specific dimension's value
    pub fn get(&self, dim: BeliefDimension) -> f32 {
        match dim {
            BeliefDimension::Confidence => self.confidence,
            BeliefDimension::Trust => self.trust,
            BeliefDimension::Relevance => self.relevance,
        }
    }

    /// Set a specific dimension
    pub fn set(&mut self, dim: BeliefDimension, value: f32) {
        let clamped = value.max(0.0).min(1.0);
        match dim {
            BeliefDimension::Confidence => self.confidence = clamped,
            BeliefDimension::Trust => self.trust = clamped,
            BeliefDimension::Relevance => self.relevance = clamped,
        }
    }

    /// Bayesian positive update — evidence supports this belief
    /// Uses Laplace smoothing: new = (current * weight + positive_strength) / (weight + 1)
    pub fn positive_evidence(&mut self, dim: BeliefDimension, strength: f32) {
        let current = self.get(dim);
        let updated = (current * 4.0 + strength) / 5.0;
        self.set(dim, updated);
    }

    /// Bayesian negative update — evidence contradicts this belief
    pub fn negative_evidence(&mut self, dim: BeliefDimension, strength: f32) {
        let current = self.get(dim);
        let updated = (current * 4.0 - strength) / 5.0;
        self.set(dim, updated);
    }

    /// Temporal decay — scores drift toward 0.5 (neutral) over time
    /// decay_rate: 0.0 = no decay, 1.0 = instant reset to 0.5
    pub fn decay(&mut self, decay_rate: f32) {
        let decay = |v: f32| v + (0.5 - v) * decay_rate;
        self.confidence = decay(self.confidence);
        self.trust = decay(self.trust);
        self.relevance = decay(self.relevance);
    }

    /// Composite belief — weighted product of all dimensions
    /// Returns 0.0-1.0. Low in any dimension drags down the whole.
    pub fn composite(&self) -> f32 {
        (self.confidence * self.trust * self.relevance).powf(1.0 / 3.0)
    }

    /// Composite with custom weights (must sum to 1.0)
    pub fn weighted_composite(&self, weights: &BeliefWeights) -> f32 {
        let c = self.confidence.powf(weights.confidence);
        let t = self.trust.powf(weights.trust);
        let r = self.relevance.powf(weights.relevance);
        (c * t * r).powf(1.0 / (weights.confidence + weights.trust + weights.relevance).max(0.01))
    }

    /// Is this belief strong enough to act on?
    /// All dimensions must exceed minimum threshold
    pub fn actionable(&self, min_confidence: f32, min_trust: f32, min_relevance: f32) -> bool {
        self.confidence >= min_confidence && self.trust >= min_trust && self.relevance >= min_relevance
    }

    /// Dominant dimension — which dimension has highest value
    pub fn dominant(&self) -> BeliefDimension {
        let mut best = BeliefDimension::Confidence;
        let mut best_val = self.confidence;
        if self.trust > best_val { best = BeliefDimension::Trust; best_val = self.trust; }
        if self.relevance > best_val { best = BeliefDimension::Relevance; }
        best
    }

    /// Weakest dimension
    pub fn weakest(&self) -> BeliefDimension {
        let mut worst = BeliefDimension::Confidence;
        let mut worst_val = self.confidence;
        if self.trust < worst_val { worst = BeliefDimension::Trust; worst_val = self.trust; }
        if self.relevance < worst_val { worst = BeliefDimension::Relevance; }
        worst
    }
}

// ── Belief Weights ───────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct BeliefWeights {
    pub confidence: f32,
    pub trust: f32,
    pub relevance: f32,
}

impl Default for BeliefWeights {
    fn default() -> Self {
        Self { confidence: 1.0, trust: 1.0, relevance: 1.0 }
    }
}

impl BeliefWeights {
    pub fn equal() -> Self { Self::default() }

    pub fn trust_heavy() -> Self {
        Self { confidence: 0.5, trust: 2.0, relevance: 1.0 }
    }

    pub fn confidence_heavy() -> Self {
        Self { confidence: 2.0, trust: 0.5, relevance: 1.0 }
    }
}

// ── Belief Store ─────────────────────────────────────────

pub struct BeliefStore {
    beliefs: HashMap<String, BeliefScore>,
    decay_per_tick: f32,
}

use std::collections::HashMap;

impl BeliefStore {
    pub fn new() -> Self {
        Self { beliefs: HashMap::new(), decay_per_tick: 0.02 }
    }

    pub fn with_decay(decay_per_tick: f32) -> Self {
        Self { beliefs: HashMap::new(), decay_per_tick }
    }

    /// Create or update a belief
    pub fn set(&mut self, key: &str, score: BeliefScore) {
        self.beliefs.insert(key.to_string(), score);
    }

    /// Get a belief score
    pub fn get(&self, key: &str) -> Option<BeliefScore> {
        self.beliefs.get(key).copied()
    }

    /// Update a single dimension of a belief
    pub fn update_dimension(&mut self, key: &str, dim: BeliefDimension, value: f32) {
        let entry = self.beliefs.entry(key.to_string()).or_insert_with(BeliefScore::default);
        entry.set(dim, value);
    }

    /// Apply positive evidence to a belief dimension
    pub fn reinforce(&mut self, key: &str, dim: BeliefDimension, strength: f32) {
        let entry = self.beliefs.entry(key.to_string()).or_insert_with(BeliefScore::default);
        entry.positive_evidence(dim, strength);
    }

    /// Apply negative evidence to a belief dimension
    pub fn undermine(&mut self, key: &str, dim: BeliefDimension, strength: f32) {
        let entry = self.beliefs.entry(key.to_string()).or_insert_with(BeliefScore::default);
        entry.negative_evidence(dim, strength);
    }

    /// Tick: decay all beliefs toward neutral
    pub fn tick(&mut self) {
        for score in self.beliefs.values_mut() {
            score.decay(self.decay_per_tick);
        }
    }

    /// Get beliefs above a composite threshold
    pub fn above_threshold(&self, min_composite: f32) -> Vec<(&str, BeliefScore)> {
        self.beliefs.iter()
            .filter(|(_, s)| s.composite() >= min_composite)
            .map(|(k, s)| (k.as_str(), *s))
            .collect()
    }

    /// Get top-N beliefs by composite score
    pub fn top_n(&self, n: usize) -> Vec<(&str, BeliefScore)> {
        let mut sorted: Vec<_> = self.beliefs.iter()
            .map(|(k, s)| (k.as_str(), *s))
            .collect();
        sorted.sort_by(|a, b| b.1.composite().partial_cmp(&a.1.composite()).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    /// Belief count
    pub fn len(&self) -> usize { self.beliefs.len() }
    pub fn is_empty(&self) -> bool { self.beliefs.is_empty() }

    /// Average composite across all beliefs
    pub fn average_composite(&self) -> f32 {
        if self.beliefs.is_empty() { return 0.0; }
        let sum: f32 = self.beliefs.values().map(|b| b.composite()).sum();
        sum / self.beliefs.len() as f32
    }
}

impl Default for BeliefStore {
    fn default() -> Self { Self::new() }
}

// ── Tests ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_score_new() {
        let b = BeliefScore::new(0.8, 0.9, 0.7);
        assert_eq!(b.confidence, 0.8);
        assert_eq!(b.trust, 0.9);
        assert_eq!(b.relevance, 0.7);
    }

    #[test]
    fn test_belief_clamping() {
        let b = BeliefScore::new(1.5, -0.3, 0.5);
        assert_eq!(b.confidence, 1.0);
        assert_eq!(b.trust, 0.0);
    }

    #[test]
    fn test_positive_evidence() {
        let mut b = BeliefScore::new(0.5, 0.5, 0.5);
        b.positive_evidence(BeliefDimension::Confidence, 1.0);
        assert!(b.confidence > 0.5);
    }

    #[test]
    fn test_negative_evidence() {
        let mut b = BeliefScore::new(0.5, 0.5, 0.5);
        b.negative_evidence(BeliefDimension::Confidence, 1.0);
        assert!(b.confidence < 0.5);
    }

    #[test]
    fn test_decay() {
        let mut b = BeliefScore::new(0.9, 0.9, 0.9);
        b.decay(0.5);
        // 0.9 + (0.5 - 0.9) * 0.5 = 0.9 - 0.2 = 0.7
        assert!((b.confidence - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_composite() {
        let b = BeliefScore::new(1.0, 1.0, 1.0);
        assert!((b.composite() - 1.0).abs() < 0.01);

        let b2 = BeliefScore::new(0.5, 0.5, 0.5);
        assert!((b2.composite() - 0.5).abs() < 0.01);

        // Low relevance drags down
        let b3 = BeliefScore::new(1.0, 1.0, 0.1);
        assert!(b3.composite() < 0.5);
    }

    #[test]
    fn test_actionable() {
        let b = BeliefScore::new(0.8, 0.9, 0.85);
        assert!(b.actionable(0.7, 0.7, 0.7));

        let b2 = BeliefScore::new(0.8, 0.9, 0.3);
        assert!(!b2.actionable(0.7, 0.7, 0.7));
    }

    #[test]
    fn test_dominant_weakest() {
        let b = BeliefScore::new(0.9, 0.3, 0.6);
        assert_eq!(b.dominant(), BeliefDimension::Confidence);
        assert_eq!(b.weakest(), BeliefDimension::Trust);
    }

    #[test]
    fn test_store_set_get() {
        let mut store = BeliefStore::new();
        store.set("tile-1", BeliefScore::new(0.8, 0.9, 0.7));
        let b = store.get("tile-1").unwrap();
        assert_eq!(b.confidence, 0.8);
    }

    #[test]
    fn test_store_reinforce_undermine() {
        let mut store = BeliefStore::new();
        store.reinforce("agent-jc1", BeliefDimension::Trust, 1.0);
        let b = store.get("agent-jc1").unwrap();
        assert!(b.trust > 0.5);

        store.undermine("agent-jc1", BeliefDimension::Trust, 1.0);
        let b2 = store.get("agent-jc1").unwrap();
        assert!(b2.trust < b.trust);
    }

    #[test]
    fn test_store_decay() {
        let mut store = BeliefStore::with_decay(0.5);
        store.set("tile-1", BeliefScore::new(1.0, 1.0, 1.0));
        store.tick();
        let b = store.get("tile-1").unwrap();
        assert!(b.confidence < 1.0);
    }

    #[test]
    fn test_store_above_threshold() {
        let mut store = BeliefStore::new();
        store.set("strong", BeliefScore::new(0.9, 0.9, 0.9));
        store.set("weak", BeliefScore::new(0.1, 0.1, 0.1));

        let above = store.above_threshold(0.7);
        assert_eq!(above.len(), 1);
    }

    #[test]
    fn test_store_top_n() {
        let mut store = BeliefStore::new();
        store.set("a", BeliefScore::new(0.3, 0.3, 0.3));
        store.set("b", BeliefScore::new(0.9, 0.9, 0.9));
        store.set("c", BeliefScore::new(0.6, 0.6, 0.6));

        let top = store.top_n(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "b");
    }

    #[test]
    fn test_store_average_composite() {
        let mut store = BeliefStore::new();
        store.set("a", BeliefScore::new(0.5, 0.5, 0.5));
        store.set("b", BeliefScore::new(1.0, 1.0, 1.0));
        let avg = store.average_composite();
        assert!(avg > 0.5 && avg < 1.0);
    }

    #[test]
    fn test_weighted_composite() {
        let b = BeliefScore::new(0.5, 1.0, 0.5);
        let weights = BeliefWeights::trust_heavy();
        let wc = b.weighted_composite(&weights);
        let ec = b.composite();
        // Trust-heavy should give more weight to trust dimension
        assert!(wc > ec); // trust=1.0 weighted more should boost composite
    }

    #[test]
    fn test_nan_hardening() {
        let mut b = BeliefScore::new(0.5, 0.5, 0.5);
        b.positive_evidence(BeliefDimension::Confidence, 0.0);
        b.negative_evidence(BeliefDimension::Trust, 0.0);
        assert!(!b.confidence.is_nan());
        assert!(!b.trust.is_nan());
        assert!(!b.relevance.is_nan());
    }

    #[test]
    fn test_rapid_evidence() {
        let mut b = BeliefScore::new(0.5, 0.5, 0.5);
        for _ in 0..20 {
            b.positive_evidence(BeliefDimension::Confidence, 1.0);
        }
        assert!(b.confidence > 0.8);
    }
}
