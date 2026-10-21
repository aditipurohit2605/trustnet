# TrustNet Pre-Registration Analysis Plan

Formal statistical evaluation plan for TrustNet Stage 12 release verification.

---

## 1. Research Objectives & Hypotheses
This analysis formalizes five pre-registered hypotheses evaluating multimodal grounding, evidence retrieval, probability calibration, symbolic reasoning, and multilingual equity.

- **Hypothesis 1 (H1: Multimodal Fusion vs. Text-Only Baseline)**:
  - *Null Hypothesis ($H_0^1$)*: Integrating visual provenance features and multimodal fusion yields no difference in Macro-F1 or Accuracy relative to the text-only baseline ($M_{\text{full}} = M_{\text{text\_only}}$).
  - *Alternative ($H_1^1$)*: Multimodal fusion alters verification performance ($M_{\text{full}} \neq M_{\text{text\_only}}$).
  - *Statistical Test*: Two-sided paired bootstrap confidence interval ($B=1000$, $\alpha=0.05$) on $\Delta \text{Macro-F1}$, and McNemar's exact paired test on classification accuracy.

- **Hypothesis 2 (H2: Evidence Retrieval Contribution)**:
  - *Null Hypothesis ($H_0^2$)*: Adding retrieved evidence passages ($\text{text\_retrieval}$) does not alter Macro-F1 compared to $\text{text\_only}$.
  - *Alternative ($H_1^2$)*: Retrieved evidence alters Macro-F1.
  - *Statistical Test*: Paired bootstrap confidence interval on difference in Macro-F1.

- **Hypothesis 3 (H3: Uncertainty Calibration & Overconfidence Bounding)**:
  - *Null Hypothesis ($H_0^3$)*: Post-hoc temperature scaling does not reduce Expected Calibration Error (ECE) relative to raw uncalibrated softmax logits ($\text{ECE}_{\text{uncalibrated}} = \text{ECE}_{\text{calibrated}}$).
  - *Alternative ($H_1^3$)*: Post-hoc temperature scaling reduces ECE ($\text{ECE}_{\text{uncalibrated}} > \text{ECE}_{\text{calibrated}}$).
  - *Statistical Test*: Paired bootstrap confidence interval on $\Delta \text{ECE} = \text{ECE}_{\text{uncalibrated}} - \text{ECE}_{\text{calibrated}}$.

- **Hypothesis 4 (H4: Grounded Symbolic Reasoning Integrity)**:
  - *Null Hypothesis ($H_0^4$)*: Source reliability weighting and symbolic conflict resolution produce zero difference in conflict attribution Macro-F1 relative to unweighted ablation.
  - *Alternative ($H_1^4$)*: Source reliability weighting alters conflict attribution Macro-F1.
  - *Statistical Test*: Exact binomial comparison and counterfactual faithfulness rate tracking.

- **Hypothesis 5 (H5: Cross-Lingual Parity)**:
  - *Null Hypothesis ($H_0^5$)*: Verification accuracy on native Hindi and Hinglish differs by more than the pre-specified equivalence margin ($\delta = 0.05$) from English.
  - *Alternative ($H_1^5$)*: Verification accuracy across languages falls within empirical equivalence bounds.
  - *Statistical Test*: Two one-sided tests (TOST) / paired bootstrap difference intervals.

---

## 2. Statistical Protocol & Invariants
1. **Paired Bootstrap Sampling**:
   - $B = 1000$ resamples drawn with replacement.
   - Percentile confidence intervals at $100(1 - \alpha)\% = 95\%$.
   - **Strict Invariance**: Comparing identical model outputs yields an exact empirical difference of $0.000$ and confidence interval $[0.000, 0.000]$.
2. **McNemar's Paired Test**:
   - Evaluates discordance in binary classification correctness between Model A and Model B.
   - When discordant pairs $n_{01} + n_{10} < 25$, the two-sided exact binomial test under $p = 0.5$ is applied.
   - Otherwise, Edwards' continuity-corrected $\chi^2$ statistic with 1 degree of freedom is used:
     $$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$
3. **Multiple Testing Correction**:
   - Holm-Bonferroni step-down adjustment controls the family-wise error rate across primary hypothesis evaluations.

---

## 3. Reporting Language & Neutrality Guidelines
- If a confidence interval encompasses zero, it is reported as:
  *"The 95% confidence interval includes zero [L, U]. Statistical evidence does not reject the null hypothesis at the 0.05 level."*
- NEVER phrase findings as *"proves no effect"*, *"completely fails"*, or *"demonstrates zero utility"*.
- Correlational associations must never be described in causal terms.
