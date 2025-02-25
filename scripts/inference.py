import onnxruntime
import numpy as np
import awkward as ak

MODEL_ONNX = "/eos/user/m/mmarcheg/ttHbb/models/multiclassifier_full_Run2_btag_LMH_8M_balance_events/spanet_output/version_3/spanet.onnx"
FILE_PARQUET = "/eos/user/m/mmarcheg/ttHbb/training_datasets/dctr/parquet/2017/output_ttHTobb_2017.parquet"

print("Loading model from", MODEL_ONNX)
session = onnxruntime.InferenceSession(
    MODEL_ONNX, 
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

print("Inputs:", [input.name for input in session.get_inputs()])
print("Outputs:", [output.name for output in session.get_outputs()])

print("Loading data from", FILE_PARQUET)
events = ak.from_parquet(FILE_PARQUET)
print(events.fields)

jet_data = ak.unzip(ak.zip({
    "pt": events.JetGood.pt,
    "eta": events.JetGood.eta,
    "sin_phi": np.sin(events.JetGood.phi),
    "cos_phi": np.cos(events.JetGood.phi),
    "btag_L": events.JetGood.btag_L,
    "btag_M": events.JetGood.btag_M,
    "btag_H": events.JetGood.btag_H
}))

met_data = ak.unzip(ak.zip({
    "pt": events.MET.pt,
    "eta": events.MET.eta,
    "cos_phi": np.cos(events.MET.phi),
    "sin_phi": np.sin(events.MET.phi)
}))

lepton_data = ak.unzip(ak.zip({
    "pt": events.LeptonGood.pt,
    "eta": events.LeptonGood.eta,
    "sin_phi": np.sin(events.LeptonGood.phi),
    "cos_phi": np.cos(events.LeptonGood.phi),
    "is_electron": ak.values_astype(abs(events.LeptonGood.pdgId) == 11, int)
}))

event_data = ak.unzip(ak.zip({
    "ht": events.events.ht
}))

jet_mask = ak.values_astype(~(ak.fill_none(ak.pad_none(events.JetGood.pt, 16, clip=True), 0) == 0), bool)
met_mask = ak.ones_like(events.MET.pt, dtype=bool)
lepton_mask = ak.ones_like(events.LeptonGood.pt, dtype=bool)
event_mask = ak.ones_like(events.events.ht, dtype=bool)

# Create dict with these keys: ['Jet_data', 'Jet_mask', 'Met_data', 'Met_mask', 'Lepton_data', 'Lepton_mask', 'Event_data', 'Event_mask']

inputs = {
    "Jet_data": ak.to_numpy(jet_data),
    "Jet_mask": ak.to_numpy(jet_mask),
    "Met_data": ak.to_numpy(met_data),
    "Met_mask": ak.to_numpy(met_mask),
    "Lepton_data": ak.to_numpy(lepton_data),
    "Lepton_mask": ak.to_numpy(lepton_mask),
    "Event_data": ak.to_numpy(event_data),
    "Event_mask": ak.to_numpy(event_mask)
}

print("Running inference")

outputs = session.run(["EVENT/tthbb"], inputs)

breakpoint()
