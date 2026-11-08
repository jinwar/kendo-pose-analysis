# Kendo Pose Analysis

Local video analysis with YOLO Pose / RTMPose comparison, contact-aligned
posture plots, and experimental shinai shaft estimation.

## Reproduce the analysis

The repository contains code, dependency pins, model-download provenance, and
reviewed strike annotations. Videos, downloaded weights, generated results,
and the Python environment stay local and are excluded from Git.

After creating the environment described below, place your original videos in
`videos/` and run these commands from the repository root:

```bash
.venv/bin/python prepare_models.py
.venv/bin/python compare.py videos/20261108_120241.mp4 --output outputs/20261108_120241_full
.venv/bin/python compare.py videos/20261108_120335.mp4 --output outputs/20261108_120335_full
.venv/bin/python detect_shinai.py
.venv/bin/python plot_strikes.py
.venv/bin/python results_index.py
.venv/bin/python -m unittest test_tracking.py test_pipeline.py
.venv/bin/python validate_results.py outputs/20261108_120241_full outputs/20261108_120335_full
```

Open `outputs/index.html` or `outputs/strike_analysis/index.html` to review the
results. Comparison output directories must be new/empty. The five strike
annotations in `strike_events.json` apply specifically to the first named video;
other recordings require their own reviewed contact frames and role IDs.

## Selected workflow: RTMPose strike alignment

YOLOX + RTMPose is the selected model for strike analysis. The original model
comparison remains available as a baseline. The strike figures use only the
saved RTMPose outputs from `20261108_120241.mp4`.

Open `outputs/strike_analysis/index.html` for the five-row overview and five
individual strike figures. Every consecutive frame from 15 frames before contact
through the contact reference is shown (~0.5 s, 16 postures per strike). The
attacker’s consecutive postures are overlaid in the same plot, with one color
per frame (blue early, red at contact). The receiver is drawn once in gray
using the contact-frame posture. Thicker lines show detected shinai shaft
segments in the corresponding frame color. All lines are solid, including uncertain
motion-blur direction estimates. The receiver's visible contact-frame shinai
segment is drawn once in gray.

Contact frames 120, 545, 810, 1087, and 1390 were selected by visual review of
consecutive source frames. They represent first apparent shinai/head contact,
with approximate uncertainty of one frame due to blur and video sampling.
These are editable manual annotations in `strike_events.json`, not automatically
detected impacts or ippon judgments. Role IDs are specified separately for each
strike because the baseline tracker changes identity across exits and crossings.

All figures share pixel scale. Within each strike a single horizontal and vertical translation
aligns the receiver's contact-frame hip position across rows; it is fixed for
the entire sequence, so the attacker's approach is preserved. No scaling,
mirroring, interpolation, temporal smoothing, or per-frame recentering is applied.
Joints below confidence 0.3 are not drawn. Raw coordinates and scores remain in
`aligned_poses.json` and `aligned_keypoints.csv`.

Regenerate shinai estimates with `.venv/bin/python detect_shinai.py`, then
figures with `.venv/bin/python plot_strikes.py`, or double-click
`Plot Strikes.command`. PNG and SVG versions are saved. The combined figure
has one row per strike; each individual figure shows the same overlaid sequence
in a larger view, with a discrete time colorbar. Contact-review image sheets are in the `review/` subfolder.

### Experimental shinai estimates

All 80 frames have a saved estimate: 60 use visible line segments selected with
wrist proximity, shaft direction, and bamboo-color filtering. For the other 20,
the detector searches for the axis of darkening relative to the first pre-strike
frame. Those results also use solid lines, but blur represents motion during an
exposure, not a single instantaneous position. No manually traced segments or
temporal interpolation are used. The heuristic is tailored to this stationary
camera and left-to-right drill; it is not a general trained shinai detector.

Endpoints show supported visible extent, **not validated handle or tip positions**.
Segments can be incomplete due to occlusion, wrapping, blur, or image boundaries.
Do not use these endpoints as full shinai length or tip-speed measurements.
The detector's quality scores are heuristic, not calibrated probabilities.

`shinai_detections.json` and `.csv` preserve estimates and method flags.
`shinai_review/strike_1_all_frames.jpg` through `strike_5_all_frames.jpg` show
every detected segment over the actual video; individual frame JPEGs are also
saved. These overlays were visually reviewed, but accuracy has not been measured
against labeled ground truth. The body coordinates and contact annotations are
unchanged by shinai detection.

Runs **YOLO26s Pose** and **YOLOX-m + RTMPose-m** on the exact same decoded
video frames. Each pipeline has an independent instance of the same simple
motion/IoU tracker. Both output the same 17 COCO body landmarks. Processing
is local; no videos are uploaded. Original videos are read only.

## View the completed comparisons

Initial runs (2026-11-08) processed every decodable frame in both supplied videos:

| Video | Frames | YOLO CPU inference fps | YOLOX + RTMPose CPU inference fps | Unique IDs (YOLO / RTMPose) |
| --- | ---: | ---: | ---: | ---: |
| `20261108_120241.mp4` | 1,541 | 31.4 | 6.4 | 7 / 9 |
| `20261108_120335.mp4` | 1,598 | 31.4 | 6.7 | 17 / 21 |

These are pipeline diagnostics on this Mac, not an accuracy ranking. RTMPose
reported higher confidence coverage, but its scores are not calibrated against
YOLO. The number of IDs shows that identity continuity needs further work for
strike analysis, especially with exits/reentries and close contact. Container
metadata lists 1,544 and 1,602 frames respectively; FFmpeg and OpenCV both decode
1,541 and 1,598. No decodable frames were sampled out.

Double-click **Open Results.command**, then choose a video. The HTML report
contains one synchronized side-by-side H.264 video, slow playback, frame-step
buttons, diagnostic metrics, and links to raw data. Open the MP4 directly in
QuickTime if preferred. HTML playback depends on your browser's local-file
support. No server, account, or internet connection is needed to view results.

Double-click **Run Comparison.command** to choose a video and process it again.
The command creates a fresh timestamped output folder. Open its `index.html`
when it finishes. The results index refreshes after every completed run.

## Run from Terminal

```bash
cd /path/to/kendo-pose-analysis

# First 10 seconds of a clip:
.venv/bin/python compare.py videos/20261108_120241.mp4 --duration 10

# A particular interval:
.venv/bin/python compare.py videos/20261108_120335.mp4 --start 15 --duration 10

# Full video (default), all frames, no sampling:
.venv/bin/python compare.py videos/20261108_120241.mp4

# Optional: accelerate only YOLO using Apple GPU. This changes the runtime comparison.
.venv/bin/python compare.py videos/20261108_120241.mp4 --device mps

.venv/bin/python -m unittest test_tracking.py test_pipeline.py
```

Both pipelines default to CPU on this Mac. YOLO uses PyTorch and RTMPose/YOLOX
use ONNX Runtime; recorded fps compares these particular implementations,
not hardware-independent model performance. Inference time includes native
preprocessing, detection, pose estimation, and model postprocessing. It excludes
warmup, video decoding, our tracker, rendering, and encoding. Execution order
alternates between models each frame. YOLO uses 640 input; YOLOX uses 640 input
and RTMPose uses 192x256 crops per person. Original-resolution frames are passed
to both pipelines, and exported coordinates are in original video pixels.

The simple shared tracker uses Hungarian assignment, predicted box position,
IoU, and center-distance gating; it retains missing tracks for about 0.5 seconds.
It has **no appearance re-identification**. A missing detection yields no pose;
the tracker does not invent coordinates. This is a controlled baseline for
comparing complete pipelines, not a claim that their native tracking systems
are equivalent. ByteTrack/BoT-SORT can be added in a later experiment.

## Outputs in each run folder

| File | Contents |
| --- | --- |
| `index.html` | Local review page with video and playback controls |
| `comparison.mp4` | YOLO left, RTMPose right; source audio omitted |
| `preview.jpg` | First processed frame with both overlays |
| `poses.jsonl` | One record per frame with both pipelines, IDs, boxes, 17 landmark coordinates and scores |
| `frames.csv` | Per-frame counts, IDs, inference times, body/wrist coverage |
| `summary.json` | Run settings, package versions, source metadata, model hashes, diagnostic summaries |

Frames use zero-based source indices. `time_seconds` comes from FFmpeg frame
presentation timestamps. `decoder_time_seconds` preserves OpenCV's raw timestamp,
which incorrectly starts negative in these MP4s. Frames are decoded sequentially
to avoid inaccurate keyframe seeks. Frame counts come from FFmpeg's decoded
frames, because container metadata can overstate them. The rendered video
uses the source's nominal average fps; variable-frame-rate timing is not exactly
reproduced, so use JSONL presentation timestamps for future movement measurements. HTML
frame stepping uses nominal fps and is for visual review.

To regenerate an overlay without rerunning inference:
`.venv/bin/python render_saved.py outputs/20261108_120241_full`

To verify data alignment and decode every output frame:
`.venv/bin/python validate_results.py outputs/20261108_120241_full`

Model-local IDs and their colors do not necessarily correspond across panels.
All detected people are retained, including bystanders or mirror reflections.
No person selection, smoothing, or interpolation is applied. Keypoints below
the display threshold are hidden in the overlay but retained in raw output.

## Interpret the comparison carefully

- Coverage counts model scores >=0.3 by default. Body coverage means the 12
  shoulder-through-ankle landmarks; wrist coverage means the two wrists.
  Both are conditional on detections, so missing people are not in the denominator.
- Confidence scores are not calibrated across models. Higher coverage is not
  evidence of better accuracy, especially for knees hidden by hakama.
- Unique IDs are not measured ID switches. More IDs can mean new people,
  false positives, reflections, or track fragmentation. Identity swaps can
  happen without an increase in ID count.
- YOLO detection threshold is 0.5 by default. RTMLib's YOLOX uses its own NMS
  and also applies an internal score cutoff at its NMS threshold (0.45).
  Scores and detections are not made equivalent just by sharing a threshold.
- RTMLib's YOLOX API discards detection scores, so those are `null` in JSONL;
  keypoint scores are still available. Empty detections are explicitly guarded
  against RTMLib's fallback of estimating a pose on the entire image.
- No shinai tracking, contact detection, strike classification, or ippon scoring
  is implemented. True pose/tracking accuracy requires manual annotations.

## Average the five strikes

After generating the individual strike plots, run `Average Strikes.command` or:

```bash
.venv/bin/python average_strikes.py
```

Open `outputs/average_strike/index.html` for the 16-frame mean overlay, each
averaged frame with variation, and a comparison against the five original poses.
PNG/SVG figures and JSON/CSV statistics are exported locally.

Each strike is translated once to place the receiver's contact-frame hip midpoint
at the origin. Matching pre-contact frames retain their actual timing; the script
checks that their timestamp differences are at most 5 ms. There is no phase
warping, scale normalization, or temporal smoothing. Joint means use confidence
weights, exclude scores below 0.3, require at least three observations, and
conservatively downweight distant estimates. The exact weights are saved in JSON.

Shinai orientations use an axial circular mean, with blurred detections weighted
at one quarter of clear detections (a heuristic). Drawn segments have a fixed
schematic length, not measured tips. All shinai lines are solid, including blurred estimates; their lower weights and
method labels remain in the exported data. Joint ellipses show weighted sample covariance with one-SD axes; shinai
fans show angular RMS spread. These describe variation, not confidence intervals.
Five repetitions cannot separate genuine technique variation from tracking error,
and manual contact uncertainty of approximately one frame is not propagated.
Averaged skeletons are not constrained to preserve anatomical bone lengths.

## Second video and two-person comparison

The reviewed contact frames for video 2 are in `strike_events_second.json`.
Person A means the attacker in video 1; Person B means the attacker in video 2.
Both use five strikes and the same 16-frame pre-contact window.

```bash
.venv/bin/python detect_shinai.py --events strike_events_second.json --output outputs/strike_analysis_second
.venv/bin/python plot_strikes.py --events strike_events_second.json --output outputs/strike_analysis_second
.venv/bin/python average_strikes.py --input outputs/strike_analysis_second/aligned_poses.json --output outputs/average_strike_second --max-time-span-ms 10
.venv/bin/python average_strikes.py
.venv/bin/python compare_average_strikes.py
```

Open `outputs/average_comparison/index.html` for side-by-side time-colored means
and 16 matching mean-posture overlays (blue A, orange B). Individual results and
variation plots are linked from that page. All shinai lines now render solid.

Video 2 has an approximately 8.3 ms timestamp difference at some matching frame
offsets, within the explicit 10 ms tolerance above; the default remains 5 ms.
Actual mean times and maximum spread are exported. No frames are interpolated or
phase-warped. The figures share pixel scale and receiver-contact hip origin, but
are not corrected for body size or perspective. Do not interpret image distances
as physical distances or differences as evidence of technique quality.

## Compare individual strikes without averaging

Run `.venv/bin/python compare_individual_strikes.py` after both individual strike
analyses are generated. Open `outputs/strike_comparison/index.html`.

The five chronological pairs (1 versus 1 through 5 versus 5) each have a
side-by-side time-colored sequence and sixteen blue/orange overlays of the
original observations. A five-row overview, PNG/SVG exports, and a JSON pairing
and timestamp audit are included. Each strike uses one fixed translation to its
receiver's contact hip; no poses are averaged or interpolated. Solid shinai lines
use the original detected segments, including blur estimates, with no invented
full weapon length. Matching timestamps are required to differ by at most 10 ms;
actual times are printed separately for A and B in each frame panel.

## Re-create the environment

The environment is already installed in `.venv`. For a fresh setup on a compatible
Mac with Python 3.9 (or adjust versions for a newer interpreter):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python prepare_models.py
```

The preparation script downloads public weights only. It stores their source
URLs and SHA-256 hashes in `models/manifest.json`. Inference uses local files.
Downloads require internet; processing after setup does not. Cache/config files
are directed to `.cache` inside the project.

## Sources and licensing

- [Ultralytics pose](https://docs.ultralytics.com/tasks/pose/): AGPL-3.0 /
  enterprise licensing for Ultralytics code/models.
- [RTMLib](https://github.com/Tau-J/rtmlib): Apache-2.0 library.
- [MMPose / RTMPose](https://github.com/open-mmlab/mmpose): model source and
  model-specific notices. Package licensing does not replace weight/dataset terms.

Videos, model weights, environments, caches, and generated outputs are excluded
by `.gitignore`. Repository: https://github.com/jinwar/kendo-pose-analysis.
