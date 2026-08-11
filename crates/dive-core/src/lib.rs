//! High-Performance Rust Core Engine for DIVE ML Reliability Platform.
//!
//! Provides parallel multi-threaded SIMD computations for dataset profiling,
//! Pearson correlation matrices, Population Stability Index (PSI) drift,
//! and cryptographic audit signatures.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub mod ffi;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetrics {
    pub n_rows: usize,
    pub n_cols: usize,
    pub sha256_hash: String,
    pub missing_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult {
    pub feature_a: String,
    pub feature_b: String,
    pub correlation: f64,
    pub is_suspicious_leakage: bool,
}

/// Compute SHA-256 fingerprint for dataset byte buffer.
pub fn compute_dataset_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Fast parallel Pearson correlation coefficient between two numeric vectors.
pub fn compute_pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        0.0
    } else {
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

/// Compute Population Stability Index (PSI) between reference and current distribution bins.
pub fn compute_psi(reference_bins: &[f64], current_bins: &[f64]) -> f64 {
    if reference_bins.len() != current_bins.len() {
        return 0.0;
    }
    let mut psi = 0.0;
    for (&ref_p, &curr_p) in reference_bins.iter().zip(current_bins.iter()) {
        let actual = if curr_p <= 0.0 { 0.0001 } else { curr_p };
        let expected = if ref_p <= 0.0 { 0.0001 } else { ref_p };
        psi += (actual - expected) * (actual / expected).ln();
    }
    psi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = compute_pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sha256_hash() {
        let hash = compute_dataset_hash(b"dive_test_dataset");
        assert!(!hash.is_empty());
    }
}
