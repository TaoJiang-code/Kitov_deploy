# Kitov Model Files

Model files are intentionally not committed here.

Copy each robot's exported ONNX files into one of these layouts:

```text
models/bumi/exported/
  FBcprAuxModel.onnx
  FBcprAuxModel.meta.json
  backward_encoder.onnx

models/g1/exported/
  FBcprAuxModel.onnx
  FBcprAuxModel.meta.json
  backward_encoder.onnx
```

The policy ONNX file name is inferred from `*.meta.json`; it does not have to
be `FBcprAuxModel.onnx` as long as the matching metadata file is named
`<policy_name>.meta.json`.

BUMI RGMT uses a single policy ONNX instead of the ONNX/backward bundle:

```text
models/bumi/rgmt/
  policy.onnx
```

The default RGMT deploy config is `configs/policy/bumi_rgmt.json`.
TorchScript `policy.pt` is still supported as a fallback when PyTorch is
installed.
