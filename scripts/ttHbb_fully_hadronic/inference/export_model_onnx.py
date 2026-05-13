import onnxruntime    # to inference ONNX models, we use the ONNX Runtime

session = onnxruntime.InferenceSession(
    "./spanet.onnx", 
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

print("Inputs:", [input.name for input in session.get_inputs()])
print("Outputs:", [output.name for output in session.get_outputs()])