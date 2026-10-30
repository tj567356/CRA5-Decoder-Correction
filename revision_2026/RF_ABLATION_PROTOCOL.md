# Prespecified parameter-matched bottleneck receptive-field ablation

Requested 2026-10-29. This is one fixed intervention, not model search.

- Original UNet2 class/forward, checkpoint and original remote model.py remain untouched.
- New DilatedUNet2 inherits the original forward and initialization; only bottleneck.0 and bottleneck.2 Conv2d dilation/padding change to (2,2) and (3,3). No parameter tensor is added or resized. Identical seeded initialization is asserted.
- Actual GPU preflight: both have 1,646,548 parameters, B=1 and8 inputs/outputs20x97x241, only these two metadata changes. Original UNet2 AST checked against original remote source and independently against Git baseline.
- Data are CRA5 normalized residual cache, not MPCA. Training2014-16, validation2017; original train statistics frozen. AdamW lr3e-4,wd1e-4,clip1,batch8,AMP,seed0,workers4,max200,patience12. Validation criterion remains original mean-batch normalized MSE including partial final batch. Model selection is saved before either test-year array is opened.
- Frozen selected checkpoint evaluated on complete2018 and complete2022 (1460 each);2022 manifest/input/target/timestamp hashes verified. No test fitting or checkpoint reselection.
- Record per-channel before/after nRMSE and gain; direct19, near13, upper6, near/upper gain ratio and near-minus-upper percentage points. Preserve per-time SSE for paired7-day bootstrap using the established script.
- Theory: exact dependency propagation through all convolution, maxpool, odd-size bilinear resize, skip union and head. Actual central support72x72 original versus96x96 dilated; phase maximum76 versus100 per axis before boundary clipping. Actual97-high domain clips the dilated height. These are potential dependencies, not empirical usage.
- Effective RF: gradient of correction f(x), never x+f(x). Eight prespecified2017 frames, all20 output channels, three positions,480 scalar probes/model. Sum squared gradients over input channels; normalize each nonzero probe before aggregation. Report zero probes,50/90/95% gradient-energy radii, mass outside the corresponding original structural support, and maps. These are local sensitivity diagnostics, not proof of full context use.
- Computation: actual training/validation wall time and steps, peak GPU allocation; both original/dilated inference profiled in the same run on identical GPU-resident2017 inputs (batch8,float32,10warmup,100timed batches,TF32/determinism flags), profiler FLOPs and model bytes. Historical original training cost will be labeled as such.
- All new outputs: research/fourcastnet/results/revision_2026/rf_ablation_d23_seed0_20261029/ locally; corresponding remote results/revision_2026 directory. Theory/preflight are separate new sibling directories. Existing results are never overwritten.
- Existing completed 3x3 affine baseline is restored to the manuscript as a **low-capacity**, non-parameter-matched control. Historical checkpoint and results are reused, not retrained or reselected. Dilated performance is not inserted into the paper before successful execution and validation.

Independent auditor: rf_ablation_audit. Limitations: single seed/region/codec; matched parameter count does not imply matched runtime or effective context, and fixed dilation does not prove a terrain mechanism.

ERF motivation: Luo et al., Understanding the Effective Receptive Field in Deep Convolutional Neural Networks, NeurIPS2016, https://proceedings.neurips.cc/paper/2016/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf . Our operational diagnostic uses squared-gradient energy and explicit quantile radii; it does not assume their theoretical distribution applies to this trained U-Net.
