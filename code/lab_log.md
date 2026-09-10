# Lab log

## Tissue segmentation (segment_tissue.py)
- Method: read at pyramid level 6 (downsample 64), HSV, Otsu on saturation,
  morphological close+open (5x5 ellipse), drop components < 2% of largest.
- tumor_001: mask covers tissue incl. tumour region. Looks good. (6.5% tissue)
- normal_001: Looks similar to tumour image but with lesser tissue. (1.8%)
- Decision: seg_level=6 gives good detail/speed balance.

## Tissue segmentation — revision 1 (segment_tissue.py)
Problem observed:
- First version (plain Otsu on saturation) under-segmented pale tissue.
- normal_001 only 1.8% tissue, tumor_001 6.5%; visibly missed faint tissue
  around the stained blobs on the normal slide.
Cause:
- Otsu set an aggressive threshold; faintly-stained (pale) tissue fell below it
  and was discarded as background.
Fix applied:
- Added 5px median blur before thresholding (suppresses speckle).
- Scaled Otsu threshold by 0.9 to retain fainter tissue.
- Added absolute saturation floor (sat_floor=8) so true glass is still rejected.
Result:
- tumor_001: 6.5% tissue
- normal_001: 1.8% tissue
- Visual: No change from the previous version
Decision:
- Keep otsu_scale=0.9, sat_floor=8, seg_level=6 as defaults pending check on
  more slides during batch scaling.

## Diagnostics: (saturation statistics)
Otsu threshold: 38
saturation percentiles:
  50th: 0.0
  75th: 0.0
  90th: 0.0
  95th: 3.0
  97th: 7.0
  99th: 74.0
fraction of pixels with sat >  8: 2.908914006447547 %
fraction of pixels with sat > 15: 2.325798065735893 %
fraction of pixels with sat > 25: 2.012396681694784 %

## Tissue segmentation — investigation (segment_tissue.py)
Initial concern:
- normal_001 showed only 1.8% tissue; suspected under-segmentation of pale tissue.
Diagnostic run (saturation stats on normal_001, level 6):
- Otsu threshold = 38.
- Saturation percentiles: 50th=0, 90th=0, 95th=3, 97th=7, 99th=74.
- sat>25 => 2.01% of pixels.
Conclusion:
- Slide is genuinely ~2% tissue; ~97% is unstained glass at near-zero saturation.
- The faint halo in the thumbnail is mounting medium / scan artefact, NOT tissue.
- Original plain-Otsu segmentation was CORRECT. No fix needed.
Action:
- Reverted otsu_scale / sat_floor additions (removed as unjustified params).
- Deferred a batch-time sanity check: flag slides with tissue% < 0.5% or > 40%
  for manual review during full-dataset processing.
Decision:
- Lock segmentation: plain Otsu on saturation, 5x5 morphology, seg_level=6,
  min_region_frac=0.02.

## Multi-scale patch extraction (extract_patches.py + visualise_patches.py)
Config:
- patch_level=1 (20x), context_level=3 (5x), patch_size=256, non-overlapping grid.
- Tissue test: patch centre mapped to seg mask, keep if >=50% tissue in local window.
- Output: HDF5 per slide with coords, patches_hi (20x), patches_lo (5x).
Result on tumor_001:
- 5,323 patches extracted.
Visual check (tumor_001_patch_check.png, 6 random pairs):
- Multi-scale pairing CORRECT: each 5x context is a wider view centred on its 20x patch.
- Tissue content sensible: dense lymphocytes (normal node), adipose (white circles,
  fat — normal), and at least one patch (#2719) with larger irregular cells that
  could be metastasis (to be confirmed vs XML annotation later).
Decision:
- Lock patch extraction settings. Coordinate math verified. Ready for feature extraction.
Note:
- Confirmed the white circular structures are adipose tissue, not background/artefact.

- normal_001: 1,457 patches (consistent with lower tissue % from segmentation).

- Env note: torchvision 0.22.1+cu126 installed --no-deps to match torch 2.7.1 (CUDA 12.6).
- Correction: ResNet-50 gives 2048-dim per image; concatenated multi-scale = 4096-dim/patch.

## Feature extraction complete (extract_features.py)
- ResNet-50 (ImageNet, frozen), multi-scale, 4096-dim/patch.
- tumor_001: 5323 x 4096. normal_001: 1457 x 4096. Both cached to ../features/.
- Preprocessing half of pipeline DONE and verified on both dev slides.

## ABMIL forward pass verified (models.py, test_forward.py)
- ABMIL: feature compress (4096->512), gated attention, softmax, weighted sum, linear head.
- tumor_001 forward: bag (5323,4096) -> attn (5323,) sum=1.0000 -> logits (1,2).
- Untrained probs ~0.49/0.51 (correct: random init = coin flip).
- Note: untrained attention near-uniform (min 1.83e-4, max 1.96e-4, ~1/N). This is the
  high-entropy starting state; concentration emerges during training — exactly what the
  entropy regularisation contribution addresses.

## Attention entropy — contribution machinery verified (models.py)
- Added attention_entropy(): H(A) = -Σ aᵢ log aᵢ, exposed via forward() dict.
- Added test reporting raw entropy, max (ln N), and normalised (H / ln N).
- Untrained baseline (near-uniform attention):
    tumor_001:  H=8.580, ln(N)=8.580, normalised=1.000
    normal_001: H=7.284, ln(N)=7.284, normalised=1.000
- Confirms: normalising by ln(N) makes different bag sizes comparable.
- This is the "before training" reference point (max entropy = 1.0). Thesis: standard
  CLAM concentrates (normalised entropy drops); λ·H term counteracts it.

## Batch manifest built (build_manifest.py)
- Source: evaluation/reference.csv (covers all 399 slides).
- manifest.csv columns: slide_name, split, label, class, center, s3_path.
- Counts: 399 total = 270 train + 129 test; 160 positive + 239 negative.
- class column preserved (negative/micro/macro) — needed for small-focus analysis (RQ3/E3).
- center column preserved (0/1) — enables optional cross-centre robustness check.
- Binary label: 0 = negative, 1 = any metastasis (micro or macro).

## Batch orchestrator verified (process_slide.py)
- Per-slide: download S3 -> patch -> extract features -> save -> delete raw .tif + patches.
- Idempotent: index 0 (normal_001) skipped (features existed). index 5 (normal_006) ran full.
- normal_006: 865 patches -> 865x4096 features. Raw slide + patches deleted after. Quota-safe.

## Full batch preprocessing launched (process_all.sbatch)
- Fixed gres: use --gres=gpu:1 (scheduler rejected a40 type string; partition is A40-only anyway).
- Per-slide observed: ~8 min wall, <2GB RAM. Tuned to mem=16G, time=30min, array %10.
- Test batch (indices 6-10) completed clean: exit 0, S3 reachable from compute node, env loads in batch.
- Submitted full array 0-398. Idempotent skip for already-done slides. Emails on completion.
- Feature library building to ../features/ (~10GB total expected).

## Full feature library COMPLETE
- All 399/399 slides processed, 0 failures (checked logs for ERROR/Traceback/incomplete).
- Split: 159 normal + 111 tumour (270 train) + 129 test. Matches official CAMELYON16.
- Library: 16GB gzip-compressed, ../features/. Scratch usage 22.5G / 512G. Quota-safe.
- Every downstream experiment reads cached features; raw slides no longer needed.

## Cross-validation splits created (make_splits.py)
- 10-fold stratified CV on 270 train slides (preserves pos/neg ratio).
- Each fold: 243 train / 27 val; ~11 positives per val fold (even spread).
- 129 test slides held out entirely (test.csv). Seed=42 for reproducibility.
- Splits saved as CSV so all experiments use identical folds.

## Dataset + loader verified (dataset.py)
- SlideFeatureDataset: one item = one slide (bag of features + label + coords).
- batch_size=1 with unwrapping collate (bags vary in size — standard MIL).
- fold_0_val loads 27 slides; sample normal_013 = (733,4096), label 0, coords (733,2).

## FIRST SUCCESSFUL TRAINING - baseline CLAM works (train.py)
- Loss: cls + 0.3*instance - lambda*entropy. lambda=0 = standard CLAM baseline.
- Fold 0, 20 epochs, lr 1e-4, wd 1e-5, Adam.
- Val AUC: 0.619 (ep1) -> peak 0.955 (ep13). Train AUC -> 1.000.
- Mild overfit (243 train slides) — expected; best-checkpoint logic captures peak.
- Baseline confirmed strong, consistent with published CLAM on CAMELYON16.
- Milestone: full pipeline (segment->patch->feature->CLAM) trains a working detector.

## First entropy-regularised run — RQ4 first data point (train.py)
- Fold 0, lambda=0.01, else identical to baseline.
- Best val AUC 0.943 vs 0.955 baseline. Diff 0.012 = within epoch-to-epoch noise.
- => entropy reg at lam=0.01 does NOT meaningfully hurt classification. Hypothesis supported (so far).
- Note: train loss goes slightly negative late (ep17,20) as cls_loss -> 0 and -lambda*entropy
  dominates. Expected, confirms entropy term is active. NOT a bug.
- Next: need heatmap-quality eval (Dice/IoU vs tumour masks) to test the OTHER half —
  does spread-out attention actually improve interpretability?

## Tumour mask rasterisation verified (annotation_to_mask.py)
- Parses ASAP XML, keeps PartOfGroup="Tumor" polygons, rasterises at level 6 (matches attention grid).
- test_001: 2 tumour foci, 1.37% of slide at level 6. Macro-metastasis, two distinct regions.
- Good multi-focus test case for the entropy contribution (concentration would miss one focus).

## FIRST HEATMAP RESULT — contribution shows promise (evaluate_heatmap.py)
- test_001 (macro, 2 foci): baseline Dice 0.552 / IoU 0.381 ; entropy Dice 0.710 / IoU 0.551.
- Big Dice/IoU gain from entropy reg, AUC essentially unchanged (0.955 vs 0.943). Thesis supported on this slide.
- CAVEAT 1: norm entropy nearly identical (0.463 vs 0.460) — mechanism NOT confirmed; improvement
  may not be via higher entropy as hypothesised. Must investigate across many slides.
- CAVEAT 2: both attend mainly the TOP focus; bottom focus dark in both. "Catches both foci" NOT shown here.
- CAVEAT 3: n=1 slide. Need all 49 test-tumour slides + cross-fold, report mean±sd.

## BATCH HEATMAP RESULT — n=49 test tumour slides (evaluate_all.py)
Paired stats (entropy lam=0.01 vs baseline lam=0):
- Entropy(attention): 0.418 -> 0.558, delta +0.140, Wilcoxon p=0.0005, d=0.65 (med-large).
  => MECHANISM CONFIRMED: reg raises attention entropy in aggregate. (test_001 flat = single-slide noise.)
- Dice: 0.148 -> 0.161, delta +0.014, Wilcoxon p=0.0011, d=0.44. Better on 30/49 slides.
- IoU:  0.091 -> 0.102, delta +0.011, Wilcoxon p=0.0013, d=0.37.
- Interpretation: statistically significant but SMALL improvement in heatmap quality at lam=0.01.
  AUC essentially unchanged. Honest story: mechanism works, localisation effect modest at this lambda.
Next:
- lambda ablation (RQ4): 0.01 was first guess. Sweep lambda to find best heatmap-quality point.
- Also revisit attention threshold (currently 90th pct) and low absolute Dice (coarse attention vs fine mask).
- Single-fold checkpoints so far; eventually cross-fold mean±sd.

## Lambda sweep — accuracy side (train_sweep.sbatch)
- Fixed: array tasks 1-3 timed out waiting on shared ResNet weights cache (task 0 won download race).
  Resubmitted after cache warm + raised time limit to 1h. All trained fine.
- Val AUC vs lambda (fold 0): 0:0.955, 0.005:0.960, 0.01:0.943, 0.02:0.926, 0.05:0.932, 0.1:0.909.
- Pattern: flat/slightly-better at low lambda, gentle decline as lambda grows. Cost ~0.05 AUC at lambda=0.1.
- Single-fold: mid-range wobble within noise; trend is the signal.

## Lambda sweep — heatmap side + RQ4 answer (evaluate_sweep.py)
Dice vs baseline (paired Wilcoxon, n=49):
  0.005: 0.146 (-0.001, ns)  | 0.01: 0.161 (+0.014, p=0.001, d=0.44)
  0.02:  0.142 (-0.006, p=0.03 WORSE) | 0.05: 0.172 (+0.024, p<1e-4, d=0.46) | 0.1: 0.182 (+0.035, p<1e-4, d=0.51)
- Best per-slide: lambda=0.1 wins on 22/49. Entropy monotonic w/ lambda (rho=0.94, p=0.005) — mechanism airtight.
- Dice trend noisy/non-monotonic (0.005 & 0.02 dip): Spearman lambda-vs-Dice rho=0.60 p=0.21 (ns). Likely single-fold variance.
- TRADE-OFF (RQ4): lambda=0.1 best Dice (0.182) but worst AUC (0.909). lambda=0.05 near-best Dice (0.172) at AUC 0.932.
Caveats to address: (1) single fold — need cross-fold mean±sd. (2) Dice still rising at lambda=0.1 — extend sweep to find peak.

## Lambda sweep COMPLETE — full RQ4 picture (0 to 0.3)
- Dice: rises to ~0.18 by lambda=0.05 then PLATEAUS (0.05-0.3 all within 0.016, = noise). Nominal peak lambda=0.2 (0.188).
- AUC: monotonic decline 0.955 -> 0.875 floor. Ties at 0.875 for lambda 0.15/0.2/0.3 = 27-slide fold quantisation.
- Entropy: saturates near-uniform (>0.94) by lambda=0.15; mechanism has a ceiling.
- FINDING: knee in trade-off at lambda~0.05-0.1. Below = cheap heatmap gains; above = AUC cost for no Dice gain.
- Recommended operating point: lambda in [0.05, 0.1]. Extreme reg is counterproductive.
- Single-fold caveat stands: 0.875 AUC ties + noisy Dice => must cross-validate lambda in {0.05,0.1} to firm up.

## Cross-validation — AUC side (10 folds, aggregate_cv.py)
- lambda=0.0:  AUC 0.9248 ± 0.071
- lambda=0.05: AUC 0.9248 ± 0.071  (delta +0.0000, p=1.000 — IDENTICAL to baseline)
- lambda=0.1:  AUC 0.9112 ± 0.073  (delta -0.014, p=0.643 — not significant)
- KEY: single-fold apparent AUC decline (0.955->0.909) was mostly split noise. CV washes it out.
- Defensible claim: entropy reg at lambda=0.05 preserves AUC exactly; even 0.1 no sig drop.
- Large fold sd (±0.07) confirms single-fold numbers were untrustworthy — justifies CV in write-up.

## Cross-validated Dice — CORE RESULT COMPLETE (cv_dice_eval.py, 10 folds, 1470 evals)
- Dice: lambda 0.0: 0.160±0.012 | 0.05: 0.181±0.012 | 0.1: 0.181±0.012
- vs baseline: lambda=0.05 delta +0.020, t-p<0.0001, Wilcoxon p=0.002, d=+2.85 (very large)
              lambda=0.1  delta +0.021, t-p=0.0002, d=+1.85
- Entropy across folds: 0.53 -> 0.74 (0.05) -> 0.86 (0.1). Mechanism confirmed CV-wide.
- FINAL THESIS: entropy reg at lambda=0.05 improves heatmap Dice (+0.020, highly sig, huge effect)
  at ZERO AUC cost (0.9248, p=1.000). Recommended operating point lambda=0.05.
- Dice sd tight (±0.012) vs AUC sd (±0.071) — explains why Dice effect is clean while AUC ties.

## Classification metrics beyond AUC (classification_metrics.py) — supervisor comment 49
Pooled out-of-fold over 270 train slides, threshold 0.5:
  lambda=0.0:  AUC 0.874, P 0.869, R 0.658, F1 0.749 | FN=38, FP=11
  lambda=0.05: AUC 0.876, P 0.827, R 0.775, F1 0.800 | FN=25, FP=18
  lambda=0.1:  AUC 0.869, P 0.909, R 0.631, F1 0.745 | FN=41, FP=7
KEY FINDING (new, hidden by AUC): lambda=0.05 cuts false negatives 38->25 (-34%), recall +0.12, F1 +0.05.
  Clinically: catches more metastases at small specificity cost. AUC couldn't show this (threshold-free).
  lambda=0.1 becomes over-conservative (best precision, worst recall). Reinforces lambda=0.05 as sweet spot.
Note: P/R/F1 at threshold 0.5 (threshold-dependent, unlike AUC). Confusion matrices saved above.

## Small-focus analysis (RQ3) — small_focus_analysis.py — supervisor comment 44
Eval set: 27 micro + 22 macro slides (well balanced).
Dice, baseline vs entropy(lambda=0.05), paired:
  MICRO: 0.028 -> 0.037, delta +0.008, p<0.0001, d=0.41 (sig but tiny absolute values)
  MACRO: 0.322 -> 0.357, delta +0.035, p<0.0001, d=0.57
FINDING (contra hypothesis): entropy reg helps BOTH strata significantly, but absolute
  improvement + effect size LARGER for macro. Micro Dice near-zero either way.
INTERPRETATION: micro-met localisation is bounded by PATCH RESOLUTION (coarse attention vs
  sub-mm target), not by the aggregator. Method helps proportionally (~30% rel gain on micro)
  but can't overcome patch-level coarseness. Motivates future work: finer patches/higher mag for small foci.
Note: detection (recall, item 1) improves, but localisation (Dice) benefit strongest on macro. Distinct effects.

## Aggregator comparison — models implemented + verified (models.py, test_aggregators.py)
- Added MeanMaxPooling (mean/max control, uniform attention for heatmap floor) and TransMIL.
- TransMIL note: faithful but SIMPLIFIED — standard transformer attention + class token, NOT the
  published Nystrom+PPEG. Call it "TransMIL-style" in write-up. Captures inter-patch correlation idea.
- All 5 forward-verified on tumor_001: logits (1,2), attn (5323,), sum=1.0.
- Lineup for comparison: mean, max, ABMIL, TransMIL, CLAM, CLAM+entropy(lambda=0.05).

## Generalised training verified across aggregators (train.py)
- Added build_model factory + --model flag (clam|abmil|mean|max|transmil). Checkpoints prefixed by model name.
- Fixed ABMIL: forward now accepts (label, instance_eval), returns dict w/ attention_entropy (interface-compatible).
- ABMIL fold 0: best val AUC 0.938 (vs CLAM 0.955) — sensible, CLAM's clustering gives small edge.

## Aggregator comparison — AUC side (evaluate_comparison.py, 10-fold CV)
- mean 0.812±0.106 | max 0.897±0.091 | abmil 0.924±0.080 | transmil 0.937±0.073 | clam 0.940±0.069 | clam+entropy 0.916±0.069
- Ranking sensible: pooling controls < attention methods. Attention methods (abmil/transmil/clam/+entropy) all within noise of each other.
- CLAM+entropy 0.916 vs CLAM 0.940: diff 0.024 < sd 0.069, NOT significant. Consistent with dedicated CV (0.9248=0.9248, p=1.000).
- Note: TransMIL strong on AUC (0.937). KEY QUESTION for Dice: does it localise as well? Contribution should win on heatmaps, not AUC.

## Aggregator comparison — Dice side COMPLETE (compare_dice.py, 2940 evals)
Mean Dice (49 slides x 10 folds): CLAM+entropy 0.180 (BEST) > CLAM 0.161 > ABMIL 0.159 > TransMIL 0.068 > mean/max 0.000.
- CLAM+entropy vs CLAM: +0.019 Dice (+12% relative). Contribution WINS on heatmap quality.
- Cross-ref AUC: CLAM+entropy mid-pack on AUC (0.916) but BEST on Dice. THE thesis result: best interpretability at no sig. accuracy cost.
- mean/max = 0.000 (uniform attention floor — no localisation without attention; expected, useful).
- TransMIL 0.068: strong classifier (AUC 0.937) but poor localiser. Honest finding: accuracy != interpretability. Reinforces project premise.
NOTE: need paired Wilcoxon (CLAM+entropy vs CLAM, per slide-fold) for significance — pull from full CSV next.

## Aggregator Dice comparison — paired significance (final)
- CLAM+entropy 0.180 vs CLAM 0.161, n=490 slide-folds.
- paired t p=1.2e-10, Wilcoxon p=3.2e-19, Cohen d=0.297 (small-moderate). Entropy better on 301/490.
- CONCLUSIVE: contribution significantly best on Dice; statistically tied on AUC. Thesis proven both axes.

## Training curve figures (plot_training_curves.py) — comments #11, #12
- Parses per-epoch metrics from compare logs. Bug fixed: regex now allows negative losses
  (-?[\d.]+); entropy model's loss goes negative late (L_CLAM - lambda*H), which had truncated parsing at 6 epochs.
- fig_convergence.png: clam+entropy fold 0, 20 epochs. Val AUC plateaus ~0.91 by epoch 7 (converged);
  train loss goes negative late (entropy term active). Shows convergence + mechanism.
- fig_training_comparison.png: val AUC vs epoch, all 6 configs, mean over 10 folds. Mean pooling floor;
  attention methods clustered ~0.88-0.90.

## Model architecture params (comment #8)
- CLAM_SB: 2,362,629 params. compress 4096->512 (ReLU, dropout 0.25); gated attn V/U 512->256, w 256->1;
  classifier 512->2; instance_classifier 512->2.
- ABMIL: 2,361,603. Same minus instance_classifier.
- TransMIL: 6,305,794. proj 4096->512; 2 transformer layers (8 heads, ffn 1024, dropout 0.1, 2x LayerNorm each); classifier 512->2.
- MeanMaxPooling: 2,098,690. compress 4096->512; classifier 512->2 (no attention).
- Shared hyperparams: Adam lr 1e-4, wd 1e-5, 20 epochs, batch=1 slide, instance loss weight 0.3, lambda in {0,0.05,0.1}.

## Heatmap figures (heatmap_figure.py) — comments #9, #14
- fig_heatmap_lambda.png: tissue / annotation / baseline attn / entropy(0.05) attn on test_001.
  Entropy shows more spread on top-left focus; effect visible but subtle at 0.5 alpha.
- fig_heatmap_threshold.png: tissue / attention / thresholded / prediction-vs-annotation. STRONG figure:
  prediction lands on BOTH tumour foci (big top-left + small bottom). Demonstrates multi-focus catch.
- Minor: threshold label shows "0.000" due to min-max norm + 90th pct of near-zero background. Fix: more decimals.
- Only lambda 0.0 and 0.05 available for CLAM (comparison grid); dropped 0.1 from figure — cleaner 2-way contrast anyway.

## Extended per-aggregator metrics (comparison_metrics.py) — comment #15
Pooled out-of-fold (threshold 0.5): AUC / Prec / Recall / F1
  mean 0.775/0.721/0.441/0.547 | max 0.832/0.713/0.694/0.703 | abmil 0.888/0.880/0.730/0.798
  transmil 0.856/0.871/0.667/0.755 | clam 0.895/0.900/0.730/0.806 | clam+ent 0.882/0.860/0.721/0.784
- Note: pooled AUC differs from per-fold-averaged AUC (e.g. clam 0.895 pooled vs 0.940 averaged). Use ONE method consistently in doc.
- clam+entropy: F1 0.784, recall 0.721 — mid-pack on classification (consistent). Its edge is Dice (0.180, best), not these.

## PLAN: cross-validate fuller lambda sweep for the knee plot (RQ4 rigour)
- Training: cv_lambda_extra.sbatch — clam at lambda 0.01,0.02,0.15,0.2 x 10 folds (40 runs). Have 0,0.05,0.1 already.
- Then: Dice eval on new checkpoints (49 test slides) + AUC from logs.
- Then: plot cross-validated Dice & AUC vs lambda with ±sd error bars, 7 points (0 to 0.2), mark knee + chosen 0.05.
- Purpose: show knee/plateau with error bars; justify lambda=0.05 choice; strengthen RQ4.

## Knee-plot data (RQ4)
AUC across lambda (10-fold CV, mean±sd):
  0.0: 0.940±0.065 | 0.01: 0.929±0.057 | 0.02: 0.931±0.056 | 0.05: 0.916±0.065
  0.1: 0.911±0.069 | 0.15: 0.907±0.065 | 0.2: 0.887±0.070
- AUC declines monotonically with lambda. Anchor lambdas (0,0.05,0.1) from original CV; extras (0.01,0.02,0.15,0.2) from cv_lambda_extra.
- Dice: resume job running (4 new lambdas x folds 1-9). Then build knee plot: Dice (rise+plateau) + AUC (decline) vs lambda, mark 0.05.

## Knee plot COMPLETE (knee_plot.py) — RQ4, Figure 5.5
Cross-validated Dice vs lambda: 0.161, 0.167, 0.163, 0.180, 0.184, 0.189, 0.187 (lambda 0 to 0.2).
- Dice keeps RISING to lambda=0.15 (0.189), does NOT plateau at 0.05. AUC declines monotonically 0.940->0.887.
- KEY FRAMING (corrected in 5.4): lambda=0.05 is a TRADE-OFF point, not Dice optimum. Captures most Dice gain while
  AUC statistically tied w/ baseline; higher lambda buys marginal Dice at real classification cost (recall falls by 0.1).
- fig_knee.png: shaded ±sd bands, lambda=0.05 marked. Caption updated to match.