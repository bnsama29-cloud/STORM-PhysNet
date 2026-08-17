# STORM-PhysNet Checkpoints

This directory contains the final trained weights for seeds 42-56 across all model configurations used in the paper.

**Loading Guidelines:**
- Always prefer the `*_best.pt` file when evaluating a model (e.g. `storm_physnet_bz_best.pt`).
- Ignore `*_last.pt` or intermediate saves if present.
- Some folders may contain multiple `*_best.pt` files (e.g. `grasp_best.pt` in addition to the main model weights). This occurs when a seed was fine-tuned for domain transfer on the GRASP dataset. Load the corresponding file for the task you are reproducing.
