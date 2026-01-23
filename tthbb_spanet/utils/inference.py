import onnxruntime as ort

# Write the code for SPANet inference here

def run_spanet_inference(cfg_spanet, jet_input, met_input, lepton_input, event_input):

    print("Loading SPANet model from", cfg_spanet["model"])
    session = ort.InferenceSession(
        cfg_spanet["model"], 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    # Run inference


    # Return output scores

