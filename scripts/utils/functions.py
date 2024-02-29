import h5py
import awkward as ak
import pandas as pd
import sys

basedir = "/eos/user/y/ymaidann/eth_project/Spanet_project/v3_sig_forTrainingDataset/"
datadir = basedir + "data/"
preddir = basedir + "predictions/"
jetsdir = basedir + "jets/"

def file_check(filename):
    try:
        file = open(filename, 'r')
        file.close()
        return True
    except FileNotFoundError:
        print(f'Could not find \'{filename}\' in given directory.')
        sys.exit(1)

def initialise_sig(file):
    """
    Prepares the data files (SPANet predictions and true data) for processing 
    and extracts jets, lepton_reco and met as Momentum4D Awkward Arrays.
    NOTE: Change file name conventions appropriately
    Returns:
        df_true (HDF5 file): data (truth)
        df_pred (HDF5 file): SPANet predictions
        jets (awkward_.MomentumArray4D): jets from data file
        lep (awkward_.MomentumArray4D, optional): reconstructed leptonic info from data file
        met (awkward_.MomentumArray4D, optional): reconstructed met info from data file
    """
    file_check(preddir + file + ".h5")
    file_check(datadir + file[:-3] + ".h5")

    df_pred = h5py.File(preddir + file + ".h5",'r')
    df_true = h5py.File(datadir + file[:-3] + ".h5",'r')

    if "matched" in file:
        file_check(jetsdir + f"all_jets_fullRun2_{file[:-11]}_v3.parquet")
        df_jets = ak.from_parquet(jetsdir + f"all_jets_fullRun2_{file[:-11]}_v3.parquet")
    else:
        file_check(jetsdir + f"all_jets_fullRun2_{file[:-3]}_v3.parquet")
        df_jets = ak.from_parquet(jetsdir + f"all_jets_fullRun2_{file[:-3]}_v3.parquet")

    if len(ak.unzip(df_jets)) == 9:
        (jets,_,_,_,_,lep, met,_,_) = ak.unzip(df_jets)
    else:
        (jets,_,_,_,_,lep,met,_) = ak.unzip(df_jets)

    jets = ak.with_name(jets, name="Momentum4D")
    lep = ak.with_name(lep, name="Momentum4D")
    met = ak.with_name(met, name="Momentum4D")
    
    return df_true, df_pred, jets, lep, met

def get_leptop_indices(df_true, df_pred, padding=False):
    """
    Args:
        df_true (HDF5 file): data (truth)
        df_pred (HDF5 file): SPANet predictions
        padding (bool, optional): keep shape of leptop_index_true after masking. Defaults to False.
    Returns:
        ind_true (ak.Array): true indices of jets associated to partons from leptonic top 
        ind_pred (ak.Array): predicted indices of jets associated to partons from leptonic top 
    """
    ind_pred = ak.Array(df_pred["TARGETS"]["t2"]["b"][()])
    ind_pred = ak.unflatten(ind_pred, ak.ones_like(ind_pred))
    ind_true = ak.Array(df_true["TARGETS"]["t2"]["b"][()])
    ind_true = ak.unflatten(ind_true, ak.ones_like(ind_true))

    # Do not take jets with index -1 
    mask = (ind_true == -1)
    ind_true = ind_true[~mask]
    if padding==True:
        ind_true = ak.pad_none(ind_true, 1)
    
    return ind_true, ind_pred

def get_hadtop_indices(df_true, df_pred, padding=False):
    """
    Args:
        df_true (HDF5 file): true data
        df_pred (HDF5 file): SPANet predictions
        padding (bool, optional): keep shape of leptop_index_true after masking. Defaults to False.
    Returns:
        ind_true (ak.Array): true indices of jets associated to partons from hadronic top 
        ind_pred (ak.Array): predicted indices of jets associated to partons from hadronic top 
    """
    t1_q1_true = ak.Array(df_true["TARGETS"]["t1"]["q1"][()])
    t1_q2_true = ak.Array(df_true["TARGETS"]["t1"]["q2"][()])
    t1_b_true = ak.Array(df_true["TARGETS"]["t1"]["b"][()])
    t1_q1_true = ak.unflatten(t1_q1_true, ak.ones_like(t1_q1_true))
    t1_q2_true = ak.unflatten(t1_q2_true, ak.ones_like(t1_q2_true))
    t1_b_true = ak.unflatten(t1_b_true, ak.ones_like(t1_b_true))
    
    t1_q1_pred = ak.Array(df_pred["TARGETS"]["t1"]["q1"][()])
    t1_q2_pred = ak.Array(df_pred["TARGETS"]["t1"]["q2"][()])
    t1_b_pred = ak.Array(df_pred["TARGETS"]["t1"]["b"][()])
    t1_q1_pred = ak.unflatten(t1_q1_pred, ak.ones_like(t1_q1_pred))
    t1_q2_pred = ak.unflatten(t1_q2_pred, ak.ones_like(t1_q2_pred))
    t1_b_pred = ak.unflatten(t1_b_pred, ak.ones_like(t1_b_pred))
 
    ind_true = ak.concatenate((t1_q1_true, t1_q2_true, t1_b_true), axis=1)
    ind_pred = ak.concatenate((t1_q1_pred, t1_q2_pred, t1_b_pred), axis=1)

    # Do not take jets with index -1 
    mask = (ind_true == -1)
    ind_true = ind_true[~mask]
    if padding == True:
        ind_true = ak.pad_none(ind_true, 3)
    
    return ind_true, ind_pred

def get_higgs_indices(df_true, df_pred, padding=False):
    """
    Args:
        df_true (HDF5 file): data (truth)
        df_pred (HDF5 file): SPANet predictions
        padding (bool, optional): keep shape of leptop_index_true after masking. Defaults to False.
    Returns:
        ind_true (ak.Array): true indices of jets associated to partons from Higgs
        ind_pred (ak.Array): predicted indices of jets associated to partons from Higgs
    """
    H_b1_true = ak.Array(df_true["TARGETS"]["h"]["b1"][()])
    H_b2_true = ak.Array(df_true["TARGETS"]["h"]["b2"][()])
    H_b1_true = ak.unflatten(H_b1_true, ak.ones_like(H_b1_true))
    H_b2_true = ak.unflatten(H_b2_true, ak.ones_like(H_b2_true))

    H_b1_pred = ak.Array(df_pred["TARGETS"]["h"]["b1"][()])
    H_b2_pred = ak.Array(df_pred["TARGETS"]["h"]["b2"][()])
    H_b1_pred = ak.unflatten(H_b1_pred, ak.ones_like(H_b1_pred))
    H_b2_pred = ak.unflatten(H_b2_pred, ak.ones_like(H_b2_pred))
 
    ind_true = ak.concatenate((H_b1_true, H_b2_true), axis=1)
    ind_pred = ak.concatenate((H_b1_pred, H_b2_pred), axis=1)
    
    # Do not take jets with index -1 
    mask = (ind_true == -1)
    ind_true = ind_true[~mask]
    if padding == True:
        ind_true = ak.pad_none(ind_true, 2)

    return ind_true, ind_pred

def delete_bug(jet_true, jet_pred, df_pred, particle):
    """
    Remove any events with assignment probability zero. In these events 
    SPANet assigns two partons to the same (sometimes imaginary) jet.
    Args:
        jet_true (awkward_.MomentumArray4D): jets from data file
        jet_pred (awkward_.MomentumArray4D): jets from prediction file
        df_pred (HDF5 file): SPANet predictions
        particle (str): h, t1 or t2
    """
    prob = ak.Array(df_pred[f"TARGETS/{particle}/assignment_probability"][()])
    jet_true = jet_true[prob > 0]
    jet_pred = jet_pred[prob > 0]
    return jet_true, jet_pred

def event_particle_matched(jet_true, jet_pred, lep=ak.Array([]), met=ak.Array([])):
    """
    Get only events where a specific event particle is matched
    Args:
        jet_true (awkward_.MomentumArray4D): jets from data file
        jet_pred (awkward_.MomentumArray4D): jets from prediction file
        lep (awkward_.MomentumArray4D, optional): reconstructed leptonic info from data file
        met (awkward_.MomentumArray4D, optional): reconstructed met info from data file
    """
    mask = (ak.num(jet_true)==ak.max(ak.num(jet_true)))
    jet_true = jet_true[mask]
    jet_pred = jet_pred[mask]
    if len(lep)>0: # if lep and met are given for leptonic top, mask them too
        lep = lep[mask]
        met = met[mask]
        return jet_true, jet_pred, lep, met
    return jet_true, jet_pred

def get_leptop_arrays(df_true, df_pred, jets, n, mode="partial"):
    """
    For efficiency plots by the number of jets in the event
    Args:
        df_true (HDF5 file): data (truth)
        df_pred (HDF5 file): SPANet predictions
        jets (awkward_.MomentumArray4D): jets from data file
        n (int): number of jets in the event
        mode (str, optional): partial mode is for a subset of events with n jets. 
        Inclusive mode is for all events. Defaults to "partial".
    Returns:
        pred_correct (awkward_.MomentumArray4D): correctly predicted jets from leptonic top
        pred_wrong (awkward_.MomentumArray4D): wrongly predicted jets from leptonic top
    """
    ind_true, ind_pred = get_leptop_indices(df_true, df_pred, padding=True)
    
    if mode == "partial":
    # i.e. consider a subset of events that have n jets
        ind_pred = ind_pred[ak.num(jets)==n]
        ind_true = ind_true[ak.num(jets)==n]
        njets = jets[ak.num(jets)==n]
    else:
        njets = jets

    jet_pred = njets[ind_pred]
    jet_true = njets[ind_true]

    mask = (jet_true == jet_pred)
    pred_correct = jet_pred[mask] # correctly predicted jets
    pred_wrong = jet_pred[~mask] # wrongly predicted jets

    pred_correct = pred_correct[ak.num(pred_correct)>0]
    pred_wrong = pred_wrong[ak.num(pred_wrong)>0]
    return pred_correct, pred_wrong

def get_higgs_arrays(df_true, df_pred, jets, n, mode="partial"):
    """
    For efficiency plots by the number of jets in the event
    Args:
        df_true (HDF5 file): data (truth)
        df_pred (HDF5 file): SPANet predictions
        jets (awkward_.MomentumArray4D): jets from data file
        n (int): number of jets in the event
        mode (str, optional): partial mode is for a subset of events with n jets. 
        Inclusive mode is for all events. Defaults to "partial".
    Returns:
        pred_correct (awkward_.MomentumArray4D): correctly predicted jets from Higgs
        pred_wrong (awkward_.MomentumArray4D): wrongly predicted jets from Higgs
    """
    ind_true, ind_pred = get_higgs_indices(df_true, df_pred, padding=True)
    
    if mode == "partial":
    # i.e. consider a subset of events that have n jets
        ind_true = ind_true[ak.num(jets)==n]
        ind_pred = ind_pred[ak.num(jets)==n]
        njets = jets[ak.num(jets)==n]
    else:
        njets = jets

    jet_true = njets[ind_true]
    jet_pred = njets[ind_pred]
        
    mask = (jet_true == jet_pred)
    pred_correct = jet_pred[mask] # correctly predicted jets
    pred_wrong = jet_pred[~mask] # wrongly predicted jets

    pred_correct = pred_correct[ak.num(pred_correct)==2]
    pred_wrong = pred_wrong[ak.num(pred_wrong)>0]
    return pred_correct, pred_wrong

def get_hadtop_arrays(df_true, df_pred, jets, n, mode="partial"):
    """
    For efficiency plots by the number of jets in the event
    Args:
        df_true (HDF5 file): data (truth)
        df_pred (HDF5 file): SPANet predictions
        jets (awkward_.MomentumArray4D): jets from data file
        n (int): number of jets in the event
        mode (str, optional): partial mode is for a subset of events with n jets. 
        Inclusive mode is for all events. Defaults to "partial".
    Returns:
        pred_correct (awkward_.MomentumArray4D): correctly predicted jets from hadronic top
        pred_wrong (awkward_.MomentumArray4D): wrongly predicted jets from hadronic top
    """
    ind_true, ind_pred = get_hadtop_indices(df_true, df_pred, padding=True)

    if mode == "partial":
    # i.e. consider a subset of events that have n jets
        ind_true = ind_true[ak.num(jets)==n]
        ind_pred = ind_pred[ak.num(jets)==n]
        njets = jets[ak.num(jets)==n]
    else:
        njets = jets

    jet_true = njets[ind_true]
    jet_pred = njets[ind_pred]
        
    mask = (jet_true == jet_pred)
    pred_correct = jet_pred[mask] # correctly predicted jets
    pred_wrong = jet_pred[~mask] # wrongly predicted jets

    pred_correct = pred_correct[ak.num(pred_correct)==3]
    pred_wrong = pred_wrong[ak.num(pred_wrong)>0]
    return pred_correct, pred_wrong

def get_predicted(particle, df_true, df_pred, jets, n, mode="partial"):
    """
    Args:
        particle (str): particle name (t1, t2, h)
        df_true (HDF5 file): data (truth)
        df_pred (HDF5 file): SPANet predictions
        jets (awkward_.MomentumArray4D): jets from data file
        n (int): number of jets in the event
        mode (str, optional): partial mode is for a subset of events with n jets. 
        Inclusive mode is for all events. Defaults to "partial".
    Returns:
        pred_correct (awkward_.MomentumArray4D): correctly predicted jets from particle
        pred_wrong (awkward_.MomentumArray4D): wrongly predicted jets from particle
    """
    if particle == "h":
        pred_correct, pred_wrong = get_higgs_arrays(df_true, df_pred, jets, n, mode)
    if particle == "t1":
        pred_correct, pred_wrong = get_hadtop_arrays(df_true, df_pred, jets, n, mode)
    if particle == "t2":
        pred_correct, pred_wrong = get_leptop_arrays(df_true, df_pred, jets, n, mode)
    return pred_correct, pred_wrong

def get_purities_by_njets(name, df_true, df_pred, jets, njet_min, njet_max):
    """
    Calculates efficiencies (purities) of t1,t2,H by the number of jets in the event.
    Returns a table with the number of correct predictions, wrong predictions and purity.
    Args:
        name (str): particle name (t1, t2, h)
        df_true (HDF5 file): data (truth)
        df_pred (HDF5 file): SPANet predictions
        jets (awkward_.MomentumArray4D): jets from data file
        njet_min (int): the minimum number of jets to consider
        njet_max (int): the maximum number of jets to consider
    Returns:
        purities (list): purity by number of jets from njet_min to njet_max
        df (pandas.DataFrame): table of results
    """
    purities = []
    df = pd.DataFrame()

    for n in range(njet_min, njet_max+1):
        pred_correct, pred_wrong = get_predicted(name, df_true, df_pred, jets, n)
        all_counts = len(pred_correct) + len(pred_wrong)
        if all_counts > 0:
            purity = len(pred_correct) / all_counts
        else:
            purity = 0
        purities.append(purity)

        row = {"correct": len(pred_correct), "wrong": len(pred_wrong), "total": all_counts, "purity": purity}
        row = pd.DataFrame(data=row, index=[len(df)+njet_min])
        df = pd.concat([df, row])

    n = "all"
    pred_correct, pred_wrong = get_predicted(name, df_true, df_pred, jets, n, mode="full")
    all_counts = len(pred_correct) + len(pred_wrong)
    purity = len(pred_correct) / all_counts
    purities.append(purity)

    row = {"correct": len(pred_correct), "wrong": len(pred_wrong), "total": all_counts, "purity": purity}
    row = pd.DataFrame(data=row, index=["inclusive"])
    df = pd.concat([df, row])

    return purities, df