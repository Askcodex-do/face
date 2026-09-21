# Phase 2.5 — Wider Face-Set Validation & Calibration Freeze

## Starting point

Work from the current `phase-2-landmark-quality` branch.

Do not:

* merge anything
* modify `main`
* modify `phase-1-foundation`
* push changes
* make production code changes unless a concrete validation defect requires a minimal fix

This is primarily a validation/audit task.

## Objective

Validate whether these current calibration values are robust beyond the four images used during tuning:

* `cubic_stretch_threshold = 1.5`
* `max_refine_shift = 0.06`

The goal is to determine whether the current face-swap engine can be frozen before GUI and packaging work.

Do not redesign the algorithm based on subjective visual preference.

---

## 1. Build the validation set

Use the largest useful collection of face images already available locally in the repository/test assets.

Do not download images from the internet.

If fewer than 8 distinct useful face images are available, create deterministic synthetic/controlled landmark fixtures to expand geometric coverage without adding dependencies.

Where possible, cover:

* frontal faces
* mild yaw
* mild roll
* different face sizes
* different image aspect ratios
* different lighting
* different contrast/background conditions
* small detected faces
* large detected faces
* source/target resolution mismatch
* fitted face scale below 1.5
* fitted face scale approximately 1.5
* fitted face scale above 1.5

Document exactly which assets/cases were actually available and tested.

Do not claim coverage for conditions that were not tested.

---

## 2. Validate `max_refine_shift = 0.06`

For every suitable test case:

Compare:

1. refinement disabled / prior-only
2. current refinement with `max_refine_shift = 0.06`

Measure where practical:

* whether refinement is actually active
* landmark displacement
* alignment residual
* temporal jitter when sequential frames are available
* whether refinement worsens alignment
* whether any landmark exceeds the configured displacement bound

Confirm:

* eye/mouth refinement behaves as intended
* jaw and nose remain untouched by the eye/mouth refinement logic
* refinement does not introduce visible or measurable instability

Verify that:

`max_refine_shift = 0`

means prior unchanged / no refinement movement, not unlimited movement.

---

## 3. Validate `cubic_stretch_threshold = 1.5`

Construct or identify cases around the boundary:

* fitted face scale clearly below 1.5
* fitted face scale approximately 1.5
* fitted face scale clearly above 1.5

Confirm that interpolation selection is based on the fitted FACE SCALE and not whole-image dimensions.

Measure where practical:

* selected interpolation method
* output detail/sharpness
* processing time
* memory behavior
* visible degradation

Do not introduce an additional pre-scaling pass.

The current single-pass interpolation approach should remain the baseline.

---

## 4. Regression suite

Run the complete existing test suite.

Record:

* total tests
* passed
* failed
* skipped
* runtime

Any failure must be investigated and classified.

---

## 5. Real-video integration test

Use an existing representative test video if available.

Verify:

* every expected frame is processed
* no frames silently pass through unchanged
* zero processing errors
* source preparation does not contaminate target-frame landmark state
* estimator reset behavior remains correct
* sequential processing remains intact
* memory remains approximately flat over the run

Record:

* frame count
* successful frames
* errors
* FPS
* peak RSS if available
* whether all output frames differ from input where replacement is expected

Do not add audio handling.

---

## 6. Classification of findings

For every observed problem, classify it as one of:

* genuine algorithmic problem
* detector limitation
* unsupported condition
* test/harness problem
* measurement limitation

Do not treat a detector limitation as an alignment regression.

Do not treat a harness error as an algorithm regression.

---

## 7. Calibration decisions

For each constant, give exactly one final status:

### `KEEP`

The current value is sufficiently supported by the broader validation set.

### `ADJUST`

Evidence supports changing the value.

If recommending `ADJUST`:

* show the evidence
* compare the proposed value against the current value
* explain the measurable improvement
* do not make the production change automatically unless the evidence is decisive

### `INSUFFICIENT DATA`

There is not enough diverse evidence to freeze or change the value.

Do not manufacture certainty.

---

## 8. Production-code discipline

Do not perform unrelated improvements.

Do not:

* redesign landmark alignment
* add multi-face support
* add audio
* add neural models
* add ONNX
* add PyTorch
* add TensorFlow/Keras
* add dlib
* add MediaPipe
* replace the sequential processing architecture
* change feathering
* enable seamless cloning by default
* add PyInstaller
* build the final GUI
* perform Windows 8.1 packaging
* change dependencies

If a concrete bug is discovered that invalidates the test, fix only the minimum necessary code and clearly document the fix.

If a production change is not necessary for validation, do not make it.

---

## 9. Required report

Create:

`reports/phase-2.5-validation.md`

The report must contain:

1. Executive summary
2. Exact branch/commit tested
3. Test assets and conditions
4. `max_refine_shift` results
5. `cubic_stretch_threshold` results
6. Boundary/threshold measurements
7. Regression test results
8. Real-video integration results
9. Memory/performance results
10. All discovered issues with classification
11. Final decision:

* `max_refine_shift = 0.06`: KEEP / ADJUST / INSUFFICIENT DATA
* `cubic_stretch_threshold = 1.5`: KEEP / ADJUST / INSUFFICIENT DATA

12. Remaining risks before engine freeze
13. Recommendation for the next development phase

Use measured evidence rather than subjective statements.

---

## Completion criteria

The task is complete when:

* the validation suite has been executed
* the two calibration constants have explicit decisions
* the complete existing test suite has passed or failures are fully explained
* a real-video integration run has been performed if a suitable video exists
* memory/performance has been checked
* the report has been created
* no unrelated production changes were made

Do not push.

Do not merge.

Stop after producing the report.
