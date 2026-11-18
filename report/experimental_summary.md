# Détection des corrélations fallacieuses dans les CNN à l'aide de l'IA explicable

Experimental summary — Waterbirds + ResNet18, baseline vs. XAI-guided background augmentation.

## 1. Research questions

- **RQ1** — Can Grad-CAM identify when a CNN relies on spurious background features?
- **RQ2** — Is background reliance associated with performance degradation when the
  background–object correlation is reversed?
- **RQ3** — Can XAI-guided targeted background augmentation reduce background reliance
  and improve worst-group accuracy?

## 2. Related work

Three references frame this study; each is used for a specific, different purpose.

**Sagawa et al. (2020), *Distributionally Robust Neural Networks for Group Shifts*
(arXiv:1911.08731).** Source of the Waterbirds dataset and of the group definition
`2*y + place` used here, and of the worst-group accuracy metric we report against
overall accuracy. The paper's central claim — that over-parameterised ERMs reach
high average accuracy while failing on atypical groups, and that strong
regularisation (L2, early stopping) is what recovers worst-group performance —
predicts exactly the pattern we measure: 87.76% overall versus 47.66% on
waterbird-on-land. Critically, their result also implies why our contribution
underperformed: the fix they identify is a change to the *training objective*,
not to the augmentation pipeline.

**Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization* (arXiv:1610.02391).** The explanation method we
apply, unmodified, to `layer4[-1]` of a ResNet18. Our metric is a derived
quantity (Grad-CAM mass outside the annotated bird box) rather than anything the
paper proposes, so the paper supports the heatmap, not the ratio. One claim in
that paper is directly relevant to RQ3 and worth noting against our negative
result: the authors argue Grad-CAM "helps achieve generalization by identifying
dataset bias". Our experiment is a direct test of that claim on Waterbirds, and
it did not hold — re-weighting by Grad-CAM attribution did not improve worst-group
accuracy.

**Xiao et al. (2021), *Noise or Signal: The Role of Image Backgrounds in Object
Recognition* (arXiv:2006.09994).** The closest prior work to RQ2, on ImageNet
instead of Waterbirds. Using a foreground/background disentanglement toolkit, they
show that (a) models reach non-trivial accuracy from background alone, (b) models
misclassify images even when the foreground is correctly classified — up to 87.5%
of the time with adversarially chosen backgrounds — and (c) *more accurate models
depend on backgrounds less*. Point (b) is the same phenomenon our Experiment C
measures, approached from the opposite direction: they perturb the background and
observe errors, we observe Grad-CAM attribution on images that already erred.
Point (c) is a prediction our data can be checked against, and it is the sharpest
limitation of our RQ2 analysis: our single-seed comparison has no accuracy
gradient across models, so we cannot test the "more accurate → less background
dependent" relationship, only the error-level association.

Note on a reference we deliberately dropped: Adebayo et al. (2018) is often cited
alongside Grad-CAM work, but its finding is that Guided Backpropagation and Guided
Grad-CAM fail the model-parameter randomisation test — plain Grad-CAM passes it.
Citing it as evidence that our explanations are unfaithful would misrepresent it.
We did not run those sanity checks; that is stated as a limitation instead.

## 3. Setup

**Hardware / software.** NVIDIA GeForce GTX 1660 SUPER (6 GB VRAM, CUDA 12.8),
PyTorch 2.11.0+cu128, torchvision 0.26.0, PyTorch-Grad-CAM 1.5.7.

Mixed precision (AMP/fp16) was tried first and **disabled**. On this GPU the first
AMP training run overflowed fp16 activations in `layer4`, producing `inf`/`NaN`
BatchNorm running statistics: train-mode accuracy rose to 98% while every
validation prediction became NaN (argmax → class 0, val accuracy exactly the
landbird base rate, 77.8%). The corrupted buffers were verified in the checkpoint
(`layer4.*.bn*.running_mean/var`). The final runs are fp32; the GTX 1660 has no
tensor cores, so AMP offered no throughput benefit. A finite-loss check now aborts
training if this ever recurs.

**Dataset.** WILDS Waterbirds v1.0 (11,788 images; official splits), binary labels
`landbird`/`waterbird`, background `land`/`water`, group = `2*y + place`.
WILDS ships no bounding boxes, so CUB-200-2011 boxes for the same filenames are
joined for the attribution metric.

| split | G0 landbird/land | G1 landbird/water | G2 waterbird/land | G3 waterbird/water | total |
|---|---|---|---|---|---|
| train | 3498 | 184 | 56 | 1057 | 4795 |
| val | 467 | 466 | 133 | 133 | 1199 |
| test | 2255 | 2255 | 642 | 642 | 5794 |

Preprocessing: resize 224×224, ImageNet normalization. Baseline training uses
random-resized-crop (scale 0.7–1.0) + horizontal flip; evaluation is deterministic.

**Training (both models).** ResNet18 ImageNet, 2-way head, AdamW (lr 1e-4, wd 1e-4),
batch 32, up to 8 epochs, early stopping (patience 3) on **validation accuracy**,
seed 42. The same split, architecture, optimizer, schedule and evaluation code are
used; only the augmentation policy differs. Baseline best epoch 4 (val 0.8699),
XAI-guided best epoch 1 (val 0.8582).

## 4. Experiment A — baseline (ERM)

| metric | value |
|---|---|
| test accuracy | **0.8776** |
| macro precision / recall / F1 | 0.8279 / 0.8078 / 0.8171 |
| F1 landbird / waterbird | 0.9223 / 0.7119 |
| group accuracy G0 / G1 / G2 / G3 | **0.9938 / 0.8727 / 0.4766 / 0.8879** |
| minority mean (G1, G2) | 0.6747 |
| worst-group accuracy | **0.4766** (G2, waterbird on land) |

Confusion matrix (rows = true, cols = predicted): `[[4209, 301], [408, 876]]`.
Training accuracy reaches 99%, i.e. the model overfits the correlated training
distribution. It handles the two majority groups well but collapses on waterbirds
placed on land, the classic shortcut signature.

Figures: `results/baseline/group_accuracy.png`, `confusion_matrix.png`,
`group_distribution.png`.

## 5. Experiment B — Grad-CAM (RQ1)

Grad-CAM on `model.layer4[-1]` for the predicted class of every test image.
The heatmap is normalized to [0, 1] and mapped to the 224×224 input.

**Background attribution ratio** = Σ CAM outside the CUB bird box / Σ CAM.
Bounding boxes are a proxy for segmentation: attribution outside a tight box can
still fall on wings/tail, and the box interior can contain background, so the ratio
is an *approximation* of background reliance, not a segmentation-based measurement.

| quantity | value |
|---|---|
| mean ratio, all test images | 0.4355 |
| G0 landbird/land | 0.4807 |
| G1 landbird/water | 0.3810 |
| G2 waterbird/land | 0.4937 |
| G3 waterbird/water | 0.4100 |

Qualitative grid (`figures/gradcam_examples.png`, also in `results/baseline/`):
rows are the four groups, columns are the highest / median / lowest ratio examples.
The highest-ratio examples are mostly **misclassified** and the CAM sits on
background or image margins; the lowest-ratio examples are correct and the CAM is
centred on the bird. Illustrative cases: a waterbird on land predicted as landbird
with attribution on foliage (ratio 1.00), and a landbird on water predicted as
waterbird with a hotspot at the right edge.

**RQ1 answer:** yes — Grad-CAM highlights background-driven decisions, and the
attribution ratio separates them numerically, with the bbox caveat above.

## 6. Experiment C — background reliance vs. errors (RQ2)

Per-image results: `results/baseline/per_image_attributions.csv` (5,794 rows).

| statistic | correct (n=5085) | incorrect (n=709) |
|---|---|---|
| mean ratio | 0.4000 | **0.6907** |
| median ratio | 0.3815 | **0.7483** |

- Mann–Whitney U = 650,311, **p = 7.0e-168**.
- Spearman(ratio, correctness) = **−0.363**, p = 8.2e-180.
- Logistic regression of error on standardized ratio: coefficient **+1.405**
  (intercept −2.654); **AUC = 0.820** for predicting errors from the ratio alone.
- Minority vs. majority mean ratio: 0.4374 vs. 0.4454 (similar overall levels,
  despite very different accuracies).

The effect is strong at the **image level**: misclassified images carry far more
attribution outside the bird. It is not monotonic at the **group level** — G0
(accuracy 0.994) and G2 (accuracy 0.477) have similar mean ratios (0.481 vs 0.494),
so the ratio describes *how* an individual prediction is made, not simply which
group it belongs to.

Figures: `figures/background_reliance.png` (box plot correct vs incorrect),
`attribution_by_group.png`, `attribution_vs_confidence.png`. Incorrect predictions
are spread across the full confidence range, including high-confidence errors that
still have high background attribution.

**RQ2 answer:** yes, there is a strong **association** (not causation): errors are
concentrated among images whose Grad-CAM mass lies mostly outside the bird.

## 7. Experiment D — XAI-guided targeted augmentation (RQ3)

**Selection (train only).** Baseline Grad-CAM on all 4,795 training images gives a
75th-percentile ratio threshold of **0.6249**; the 1,199 training images above it
are selected. Test images, test labels and test explanations are never used.

**Intervention.** Selected images receive a stronger background-altering policy
while the bird is (approximately) preserved: RandomResizedCrop scale 0.45–1.0,
colour jitter (0.5/0.5/0.5/0.15), horizontal flip, random erasing (p = 0.35,
2–12% of area). Non-selected images keep the baseline policy. Identical
architecture/optimizer/schedule/seed otherwise.

Training stopped early at epoch 4; the best average-validation checkpoint was
epoch 1 (val accuracy 0.858; validation worst-group 0.376). Note that epoch 2 had
a better validation worst-group (0.556) but a slightly lower average accuracy, and
was not selected because checkpoint selection follows average validation accuracy
for both models.

| metric | baseline | XAI-guided | Δ |
|---|---|---|---|
| overall accuracy | 0.8776 | 0.8776 | 0.0000 |
| worst-group accuracy | **0.4766** | **0.4003** | **−0.0763** |
| minority mean accuracy | 0.6747 | 0.6507 | −0.0240 |
| G0 landbird/land | 0.9938 | 0.9965 | +0.0027 |
| G1 landbird/water | 0.8727 | 0.9011 | +0.0284 |
| G2 waterbird/land | 0.4766 | 0.4003 | −0.0763 |
| G3 waterbird/water | 0.8879 | 0.8551 | −0.0328 |
| macro F1 | 0.8171 | 0.8090 | −0.0081 |
| mean background ratio (all) | 0.4355 | 0.4548 | +0.0192 (worse) |
| ratio, correct predictions | 0.4000 | 0.4225 | +0.0226 (worse) |
| ratio, incorrect predictions | 0.6907 | 0.6860 | −0.0047 |
| ratio, minority groups | 0.4374 | 0.4574 | +0.0200 (worse) |
| ratio, majority groups | 0.4454 | 0.4651 | +0.0197 (worse) |

The RQ2 association persists in the modified model (Spearman −0.340, AUC 0.800).

**RQ3 answer: no.** This implementation did **not** reduce background reliance or
improve worst-group accuracy. Overall accuracy was unchanged (both models classify
5,085/5,794 correctly), landbird-on-water improved (+2.8 pp), but waterbird-on-land
dropped by 7.6 pp and waterbird-on-water by 3.3 pp. Mean background attribution
increased slightly. The result is reported as a negative/partial outcome; a positive
conclusion is not forced.

**Likely reasons.** (i) Aggressive random-resized crops can remove the bird, which
may *increase* reliance on remaining context rather than teach object focus.
(ii) The global CAM ratio selects images where the bird is small or ambiguous, not
necessarily images driven by a *background class cue*. (iii) Giving much stronger
augmentation to only 25% of samples creates an inconsistent training distribution.
(iv) Short training with selection on average validation accuracy (not worst-group)
favours the early XAI checkpoint, which had the weakest validation worst-group score.

## 8. Limitations

- Bounding boxes approximate the bird region; the ratio is not segmentation-based.
- Grad-CAM explains the predicted class at a coarse spatial resolution; ratios are
  comparable across images but not an absolute measure of "percent background use".
- Observational association: high background attribution and errors co-occur; this
  does not establish that attribution causes the error.
- One seed, one short training schedule, no hyperparameter search; both models
  overfit the training split. Groups 1/2 are tiny in training (184/56), so
  validation-group estimates are noisy.
- Checkpoint selection used average validation accuracy, which is not group-robust.
- fp16/AMP was abandoned for numerical stability on this GPU (documented above).

## 9. Reproducing

```bash
python run_experiments.py                     # full pipeline
python run_experiments.py --steps baseline    # or individual stages
python run_experiments.py --steps gradcam,rq2
python run_experiments.py --steps xai
python run_experiments.py --steps compare
```

Outputs: `results/`, `checkpoints/`, `figures/`; interactive walkthrough in
`notebooks/experiments.ipynb`. Seeds, thresholds and hyperparameters are in
`config.py`.

## 10. Future work

Background-only augmentation using CUB part/segmentation masks (paste a different
background outside the bird mask); selecting training images by *foreground* vs
*background* attribution contrast rather than a global ratio; group-balanced
sampling or group-DRO; checkpoint selection on worst-group validation accuracy;
multiple seeds; longer schedules with cosine decay.
