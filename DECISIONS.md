# Modeling Decisions Log

## REJECTED: explicit cell-level feature families (2026-06-08)
Tested leak-safe: group_baselines, recent_stats, slope_features (+5-7 cols, 92->99 feat).
Full-mode A/B, both arms 3-seed, ran twice.
- Scored OOF MAE: baseline 2.0733 -> treatment 2.11-2.13 (+2% WORSE), all 3 categories worse.
- OOF-vs-probe gap widened +30% (leak/instability guard fired).
- cell_expanding_mean ranked high (#3) but fold-to-fold unstable, not signal.
- Runtime +84%.
The old 251-feature run's high importance for these was partly LEAK-INFLATED
(old recent4_vs_hist_ratio used all-train windows). DO NOT re-add from importance charts.

## Levers tested and dead (this data is noise-floored; scored error ~90% scatter):
ensemble (correlated ~0.97, <0.2% gain), more seeds (saturated at 0.7-1.2%),
robust loss Huber/Fair (rejected), transform none/sqrt/log1p (neutral ~1%),
Ridge (1.19-1.34, loses to CatBoost on matched metric), explicit features (above).
Real lever = recursive forecasting (scored ~1.9 -> ~1.32). Generalization merit =
clean CV + honest contracts, not more modeling.

group-relational features (rank/peer/gap on high-card jurisdiction): KEPT, default on.
A/B 1 run: OOF neutral (2.049->2.044), recursive -2.8% (1.369->1.330), all_drugs tail
-1.4% + val max ->59.7, opioids +2.9% worse. Leak-safe (gap conservative, verify 0).
Marginal keep on recursive+tail; OOF within noise. Not confirmed across multiple runs.