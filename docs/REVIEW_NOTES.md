# Methodological notes

Caveats that matter for interpreting or extending these results. They are stated here rather than buried so that anyone reusing the protocol inherits the right assumptions.

---

## 1. Absolute accuracies are low by construction

Every model is used as a **frozen feature extractor** and nothing is fine-tuned. The probes therefore measure the linear separability of fixed representations, not the ceiling of a trained model — the same cancer baseline reaches 0.673 balanced accuracy when fine-tuned end to end, versus 0.62 as a frozen probe in-domain.

Do not read a probe accuracy of 0.21 as "this model is 21% accurate at dermatology". Read it as "a linear readout of these frozen features separates these categories this well". The argument rests on relative drops across domains and on the label-free analysis (§3), which requires no classifier at all.

Accuracies are also **not chance-normalized across tasks of differing class count**: in-domain is 8 classes, SCIN categories are 7, SCIN fine-grained is 70, DDI is binary. The kNN purity *lift* in §3 subtracts the chance floor and is the more directly comparable quantity.

## 2. The tone gap is not distinguishable from zero

The DDI tone gap is `max - min` of accuracy across three Fitzpatrick groups. Two problems compound:

1. **The statistic is biased upward.** The maximum of three noisy estimates exceeds the minimum even when all three groups share an identical true accuracy.
2. **The bootstrap interval cannot contain zero.** Resampling each group and taking `max - min` yields a non-negative quantity whose lower percentile is essentially never zero, so the interval is *not* a test against a null of no gap.

`python -m dermgap.null_calibration` quantifies this. At DDI's actual test-split sizes (55 / 65 / 77 images per group) and a plausible shared accuracy of 0.75, the expected gap **with no tone effect whatsoever** is ≈0.09, with a 95% range of ≈0.02–0.20:

| Model | Observed gap | Percentile under the null | |
|---|---|---|---|
| Cancer baseline | 0.143 | 86th | within null range |
| DermLIP | 0.136 | 83rd | within null range |
| MONET | 0.096 | 58th | within null range |
| DINOv3 | 0.175 | 95th | within null range |

All four fall inside the null range; MONET's is almost exactly the null expectation. The defensible claim is that **the tone effect is not resolvable at DDI's sample size**, not that a gap of 0.10–0.18 was measured. This is the conservative reading, and it strengthens rather than weakens the paper's conclusion that distribution shift dominates.

Note also that the tone gap uses **plain accuracy** per group, not balanced accuracy, because several tone groups contain only one class in the test split. The overall DDI number *is* balanced accuracy. These are different metrics and should not be compared directly.

## 3. kNN purity: what is and is not controlled

Purity is the fraction of each image's k=10 cosine nearest neighbours sharing its label, minus the random-neighbour floor `sum_i p_i^2`. The subtraction means class imbalance cannot inflate the metric. Two things remain uncontrolled:

- **Embedding dimensionality.** The four models produce 2048-, 512-, 1024- and 768-dimensional features. Neighbourhood structure in high dimensions is not directly comparable to low dimensions.
- **Class count.** Figure 2 compares in-domain purity at **8 classes** against SCIN purity at **47 classes**. Lift is chance-corrected, so the comparison is defensible, but the class counts differ by a factor of six.

The matched-granularity check is SCIN at the **category** level (5–7 classes, close to in-domain's 8). It preserves the ordering — cancer baseline lowest, DermLIP highest, general-vision control in between — with smaller magnitudes. Report both if space allows; report the granularity either way.

## 4. Imaging modality is confounded with disease distribution

HAM10000 and ISIC 2019 are predominantly **dermoscopic**. SCIN is **clinical photography** (smartphone images crowdsourced via search ads). So the shift from source to SCIN is simultaneously a disease shift and a modality shift, and a critic can reasonably ask which one causes the collapse.

Three pieces of evidence bear on this:

1. **DDI is also clinical photography**, and the cancer baseline still performs reasonably there (0.670 balanced accuracy on binary malignant). Modality change alone does not destroy it.
2. **The cancer baseline scores 0.000 recall on SCIN's neoplastic category** — the one category it was actually trained for, and the one where modality is the only barrier. Even the familiar disease class fails.
3. Rikhye et al. (eBioMedicine, 2025) report independently that condition distribution, rather than image capture mode, drove errors in their teledermatology deployment.

Point 1 is the direct control and belongs in the paper. Point 2 cuts both ways and should be presented honestly: it shows the failure is not merely a missing output head, but a modality-only explanation would also predict it.

## 5. Grouping keys are not always patients

The split grouping key is the finest identifier each dataset distributes:

| Dataset | Key | Granularity |
|---|---|---|
| HAM10000 | `lesion_id` | lesion — no patient ID is distributed |
| ISIC 2019 | `lesion_id`, falling back to image id | lesion |
| DDI | `DDI_ID` | **image** (656 IDs for 656 images) |
| SCIN | `case_id` | case (one crowdsourced submission) |

Grouping prevents the leakage it can prevent — multiple images of one lesion or one SCIN case never straddle the split. But DDI **cannot** be split at the patient level with the released metadata, so its split is effectively a random image-level split. Describe it that way.

## 6. SCIN data cleaning drops a large share of cases

SCIN ships `weighted_skin_condition_label` as a dict of `{condition: weight}`; this pipeline takes the arg-max. Cases whose dict is empty are dropped: **5,033 → 3,061 cases, a 39.2% reduction**, yielding 6,517 images after one missing file is removed.

That is a substantial filter and it is not random — cases without a confident dermatologist label are plausibly the ambiguous ones. The retained subset is therefore easier than SCIN as a whole, which if anything makes the observed collapse on it more striking. Any reuse of these numbers should carry the caveat.

Fine-grained probes additionally drop classes with fewer than 10 images (70 of 211 labels survive, 6,033 images); purity uses a threshold of 20 for SCIN, 15 for DDI, 20 in-domain.

## 7. Standardization hurt three of four models

Feature standardization before the probe was pre-specified, but it was not helpful:

| Model | Raw | Standardized | Change |
|---|---|---|---|
| Cancer baseline | 0.214 | 0.207 | −0.007 |
| DermLIP | 0.421 | 0.359 | **−0.063** |
| MONET | 0.332 | 0.342 | +0.010 |
| DINOv3 | 0.352 | 0.311 | −0.041 |

This matters for one specific claim. "The few-shot probe matches or exceeds the full probe" holds against the **standardized** full probe (DermLIP 0.411 few-shot vs 0.359 full). Against **raw** features, DermLIP's full probe (0.421) still edges out its few-shot probe (0.411). The qualitative conclusion — roughly ten labeled examples per condition recovers most of the attainable performance — survives either way, but the comparison should say which variant it uses.

The purity-versus-recovery correlation is also mildly variant-dependent; `dermgap.evaluate` reports it against both the standardized and raw probes.

## 8. Preprocessing deviates slightly from official transforms

To reproduce the paper, images are squash-resized to 224×224 for the cancer baseline, MONET and DINOv3. The canonical transforms are resize-shortest-side plus centre crop. DermLIP alone uses its own `open_clip` transform.

This is a mild mismatch that could disadvantage MONET and DINOv3 relative to DermLIP. `extract_features.py --official-preprocess` switches MONET and DINOv3 to their canonical transforms as a robustness check.

## 9. Possible pretraining contamination

DDI has been public since 2022 and SCIN since 2024. DermLIP (Derm1M) and MONET were trained on large clinical corpora assembled from the literature and the web, so neither can be assumed disjoint from these evaluation sets. Neither model's training data is fully enumerable, so contamination cannot be ruled out. The comparison that is safe from this concern is the **cancer baseline versus everything else**, since its training data is fully known.

## 10. Small n for model-level statistics

The correlation between latent structure and recoverable performance (r = 0.90) is computed over **four models**. It is descriptive, not inferential, and no p-value should be attached to it. It also correlates fine-grained purity (47 classes) against a category-level probe (7 classes); the granularity mismatch is deliberate but should be stated.
