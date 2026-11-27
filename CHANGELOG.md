# Changelog

## Unreleased

### Training and data tools

- Added local asset discovery and snapshot extraction, including source metadata, visual audits and an optional PSD2Live teacher adapter. PSD2Live outputs are candidate supervision in explicitly enabled experiments.
- Added parameter-specific Real Rig heads, guarded and curriculum training variants, full-body Extreme variants and a unified rig research trainer.
- Added bounded replay experiments for blink deformation, learning rate, gradient balancing and projection, and right-eye gate learning. Eligible milestones can be saved separately with model and optimizer state.
- Added local dashboards, thermal supervision, and interruption checks during texture and snapshot loading.

### Diagnostics and preview fixes

- Added three-way comparisons of the original rig, baseline and candidate with shared pose and framing, plus moving and static error breakdowns and drawable isolation.
- Textured comparisons use Cubism Core render orders while preserving the stored geometry and evaluator. Texture compositing remains a preview of selected rig attributes.
- Added geometry provenance checks, identity-delta verification tooling and a Real Rig to Unified conversion audit.

### Experimental results and limits

- Short replay runs with gradient projection produced eligible intermediate checkpoints, but later checkpoints could regress on older exam poses. These results did not meet mastery.
- Learning the right-eye gate alongside its head modestly improved affected-region precision and static RMS at an eligible milestone. Static leakage and large deformation errors remain, and both branches failed the old-exam maximum error requirement at the last milestone.
- UnifiedRig, the PSD2Live adapter and the training variants are research tools. The release does not include model assets, textures, datasets, SDK binaries, generated comparisons or checkpoints.
