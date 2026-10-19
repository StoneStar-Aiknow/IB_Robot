"""Build the GraspGen-X diffusion loop as one Ascend offline model.

`export_onnx` reads the loop's constants out of a bundle's generator and
assembles the unrolled graph; `onnx2om` compiles it with ATC. Both run on the
device: reading the constants needs Torch-NPU, and the graph has to be
compiled for the SoC that will serve it.
"""
