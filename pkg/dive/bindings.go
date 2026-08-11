// Package dive provides Go SDK bindings and high-speed dataset reliability utilities.
package dive

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
)

// DatasetMetrics holds Go dataset summary metrics.
type DatasetMetrics struct {
	Rows       int     `json:"n_rows"`
	Cols       int     `json:"n_cols"`
	SHA256Hash string  `json:"sha256_hash"`
	MissingPct float64 `json:"missing_pct"`
}

// ComputeSHA256 returns hex-encoded SHA-256 fingerprint of input data bytes.
func ComputeSHA256(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

// ComputePearsonCorrelation calculates Pearson correlation between two float64 slices in pure Go.
func ComputePearsonCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0.0
	}
	n := float64(len(x))
	var sumX, sumY float64
	for i := 0; i < len(x); i++ {
		sumX += x[i]
		sumY += y[i]
	}
	meanX := sumX / n
	meanY := sumY / n

	var cov, varX, varY float64
	for i := 0; i < len(x); i++ {
		dx := x[i] - meanX
		dy := y[i] - meanY
		cov += dx * dy
		varX += dx * dx
		varY += dy * dy
	}

	if varX == 0.0 || varY == 0.0 {
		return 0.0
	}
	return cov / (math.Sqrt(varX) * math.Sqrt(varY))
}

// ComputePSI calculates Population Stability Index between reference and current bin frequencies.
func ComputePSI(refBins, currBins []float64) float64 {
	if len(refBins) != len(currBins) || len(refBins) == 0 {
		return 0.0
	}
	var psi float64
	for i := 0; i < len(refBins); i++ {
		actual := currBins[i]
		if actual <= 0 {
			actual = 0.0001
		}
		expected := refBins[i]
		if expected <= 0 {
			expected = 0.0001
		}
		psi += (actual - expected) * math.Log(actual/expected)
	}
	return psi
}

// PrintBanner outputs branded DIVE Go terminal header.
func PrintBanner(title, subtitle string) {
	fmt.Println()
	fmt.Printf("⚡ \033[1;36m%s\033[0m\n", title)
	if subtitle != "" {
		fmt.Printf("   \033[0;37m%s\033[0m\n", subtitle)
	}
	fmt.Println()
}
