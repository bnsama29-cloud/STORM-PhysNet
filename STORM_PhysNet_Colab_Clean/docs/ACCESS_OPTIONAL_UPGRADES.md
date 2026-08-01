# Optional Access upgrades (acceptance-oriented)

Keep the core story. Add only items that strengthen reviewer confidence.

## 1. Compute-cost table (high value, low risk)

| Model | Parameters | ms / batch 32 |
|-------|------------|---------------|
| Transformer | ~845k | (from compute_cost.csv) |
| STORM-BzGate | ~343k | (from compute_cost.csv) |

Text: STORM is smaller than the matched Transformer while improving storm PE and 45 min stability.

## 2. Short statistical summary (already partly in text)

One sentence is enough: mean ΔPE_all ≈ 0.021, Cohen d ≈ 0.40, n=3 seeds, sign-flip not significant; emphasize storm PE and 45 min reliability.

## 3. Data-availability sentence

GOES-15 EPEAD and OMNI from public archives; GRASP used under the problem-setting terms; code link.

## 4. Do not add as main claims

- MC dropout as calibrated uncertainty
- HSS/POD from broken runs
- Hybrid gate failure as a highlight
- Inflated PE from other datasets/horizons

## 5. Figure that help acceptance

Already strong: architecture, horizon PE, ablation, tau, gate, importance, residual, GRASP, one dashboard.
