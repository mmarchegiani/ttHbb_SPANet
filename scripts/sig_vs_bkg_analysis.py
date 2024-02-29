# Generates plots of fully matched ttH signal, inclusive ttH signal, and ttH backgrounds.
# 2023-10-26 22:28:51
# Yuliia Maidannyk yuliia.maidannyk@ethz.ch

import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep

from utils.kinematics import *
from utils.functions import *

hep.style.use(hep.style.ROOT)
vector.register_awkward()
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
mpl.rcParams['grid.alpha'] = 0.2
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 24

def get_matched(jets):
    """
    Get only fully matched jets
    """
    higgs = jets[jets.prov == 1]
    mask_match = ak.num(higgs) == 2

    w_or_t_jets = jets[(jets.prov == 5)|(jets.prov == 2)]
    mask_match = mask_match & (ak.num(w_or_t_jets) == 3)

    lep_top = jets[jets.prov == 3]
    mask_match = mask_match & (ak.num(lep_top) == 1)

    jets = jets[mask_match]
    return jets

# Directory to save the plots
figdir = "/eos/user/y/ymaidann/eth_project/Spanet_project/plots/"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ SIGNAL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nAnalysing signal...\n")

sig1 = "ttHTobb_forTraining_2017_matched_v9"
sig1_true, sig1_pred, sig1_jets, _, _ = initialise_sig(sig1)
sig2 = "ttHTobb_forTraining_2018_matched_v9"
sig2_true, sig2_pred, sig2_jets, _, _ = initialise_sig(sig2)
sig3 = "ttHTobb_forTraining_2016_PostVFP_matched_v9"
sig3_true, sig3_pred, sig3_jets, _, _ = initialise_sig(sig3)
sig4 = "ttHTobb_forTraining_2016_PreVFP_matched_v9"
sig4_true, sig4_pred, sig4_jets, _, _ = initialise_sig(sig4)

sig1_jets = get_matched(sig1_jets)
sig2_jets = get_matched(sig2_jets)
sig3_jets = get_matched(sig3_jets)
sig4_jets = get_matched(sig4_jets)

# Get jets coming from the Higgs
sig1_ind_true, sig1_ind_pred = get_higgs_indices(sig1_true, sig1_pred)
sig1_h_true = sig1_jets[sig1_ind_true]
sig1_h_pred = sig1_jets[sig1_ind_pred]
sig2_ind_true, sig2_ind_pred = get_higgs_indices(sig2_true, sig2_pred)
sig2_h_true = sig2_jets[sig2_ind_true]
sig2_h_pred = sig2_jets[sig2_ind_pred]
sig3_ind_true, sig3_ind_pred = get_higgs_indices(sig3_true, sig3_pred)
sig3_h_true = sig3_jets[sig3_ind_true]
sig3_h_pred = sig3_jets[sig3_ind_pred]
sig4_ind_true, sig4_ind_pred = get_higgs_indices(sig4_true, sig4_pred)
sig4_h_true = sig4_jets[sig4_ind_true]
sig4_h_pred = sig4_jets[sig4_ind_pred]

sig1_matched_m = mass_h(sig1_h_pred)
sig2_matched_m = mass_h(sig2_h_pred)
sig3_matched_m = mass_h(sig3_h_pred)
sig4_matched_m = mass_h(sig4_h_pred)
sig_matched_m = np.hstack((sig1_matched_m, sig2_matched_m, sig3_matched_m, sig4_matched_m))

plt.figure() 
plt.hist(sig_matched_m, bins=100, histtype='step', label="predicted signal", range=(0,300))
plt.xlim(0,300)
plt.xlabel(r"$m_{bb}$")
plt.legend()
plt.title("Fully matched events: signal", pad=15)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"sig_matched_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

sig1 = "ttHTobb_forTraining_2017_v9"
sig1_true, sig1_pred, sig1_jets, _, _ = initialise_sig(sig1)
sig2 = "ttHTobb_forTraining_2018_v9"
sig2_true, sig2_pred, sig2_jets, _, _ = initialise_sig(sig2)
sig3 = "ttHTobb_forTraining_2016_PostVFP_v9"
sig3_true, sig3_pred, sig3_jets, _, _ = initialise_sig(sig3)
sig4 = "ttHTobb_forTraining_2016_PreVFP_v9"
sig4_true, sig4_pred, sig4_jets, _, _ = initialise_sig(sig4)

# Get jets coming from the Higgs
sig1_ind_true, sig1_ind_pred = get_higgs_indices(sig1_true, sig1_pred)
sig1_h_true = sig1_jets[sig1_ind_true]
sig1_h_pred = sig1_jets[sig1_ind_pred]
sig2_ind_true, sig2_ind_pred = get_higgs_indices(sig2_true, sig2_pred)
sig2_h_true = sig2_jets[sig2_ind_true]
sig2_h_pred = sig2_jets[sig2_ind_pred]
sig3_ind_true, sig3_ind_pred = get_higgs_indices(sig3_true, sig3_pred)
sig3_h_true = sig3_jets[sig3_ind_true]
sig3_h_pred = sig3_jets[sig3_ind_pred]
sig4_ind_true, sig4_ind_pred = get_higgs_indices(sig4_true, sig4_pred)
sig4_h_true = sig4_jets[sig4_ind_true]
sig4_h_pred = sig4_jets[sig4_ind_pred]

# Delete the bug where b1,b2 indices are the same
sig1_h_true, sig1_h_pred = delete_bug(sig1_h_true, sig1_h_pred, sig1_pred, "h")
sig2_h_true, sig2_h_pred = delete_bug(sig2_h_true, sig2_h_pred, sig2_pred, "h")
sig3_h_true, sig3_h_pred = delete_bug(sig3_h_true, sig3_h_pred, sig3_pred, "h")
sig4_h_true, sig4_h_pred = delete_bug(sig4_h_true, sig4_h_pred, sig4_pred, "h")

sig1_m = mass_h(sig1_h_pred)
sig2_m = mass_h(sig2_h_pred)
sig3_m = mass_h(sig3_h_pred)
sig4_m = mass_h(sig4_h_pred)
sig_m = np.hstack((sig1_m, sig2_m, sig3_m, sig4_m))

plt.figure() 
plt.hist(sig_m, bins=100, histtype='step', label="predicted signal", range=(0,300))
plt.xlim(0,300)
plt.xlabel(r"$m_{bb}$")
plt.legend()
plt.title("Inclusive events: signal", pad=15)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"sig_incl_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

plt.figure() 
plt.hist(sig1_m, bins=100, density=True, histtype='step', range=(0,300), label="ttHTobb_2017")
plt.hist(sig2_m, bins=100, density=True, histtype='step', range=(0,300), label="ttHTobb_2018")
plt.hist(sig3_m, bins=100, density=True, histtype='step', range=(0,300), label="ttHTobb_2016_PostVFP")
plt.hist(sig4_m, bins=100, density=True, histtype='step', range=(0,300), label="ttHTobb_2016_PreVFP")
plt.xlim(0,300)
plt.xlabel(r"$m_{bb}$")
plt.legend(bbox_to_anchor=(1,1))
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"norm_sig_split_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ TTbb BACKGROUND ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nAnalysing TTbb background...\n")

bkg1 = "TTbbSemiLeptonic_Powheg_2016_PostVFP_v9"
bkg1_true, bkg1_pred, bkg1_jets, _, _ = initialise_sig(bkg1)
bkg2 = "TTbbSemiLeptonic_Powheg_2016_PreVFP_v9"
bkg2_true, bkg2_pred, bkg2_jets, _, _ = initialise_sig(bkg2)
bkg3 = "TTbbSemiLeptonic_Powheg_2017_v9"
bkg3_true, bkg3_pred, bkg3_jets, _, _ = initialise_sig(bkg3)
bkg4 = "TTbbSemiLeptonic_Powheg_2018_v9"
bkg4_true, bkg4_pred, bkg4_jets, _, _ = initialise_sig(bkg4)

# Get jets coming from the Higgs
bkg1_ind_true, bkg1_ind_pred = get_higgs_indices(bkg1_true, bkg1_pred)
bkg1_h_true = bkg1_jets[bkg1_ind_true]
bkg1_h_pred = bkg1_jets[bkg1_ind_pred]
bkg2_ind_true, bkg2_ind_pred = get_higgs_indices(bkg2_true, bkg2_pred)
bkg2_h_true = bkg2_jets[bkg2_ind_true]
bkg2_h_pred = bkg2_jets[bkg2_ind_pred]
bkg3_ind_true, bkg3_ind_pred = get_higgs_indices(bkg3_true, bkg3_pred)
bkg3_h_true = bkg3_jets[bkg3_ind_true]
bkg3_h_pred = bkg3_jets[bkg3_ind_pred]
bkg4_ind_true, bkg4_ind_pred = get_higgs_indices(bkg4_true, bkg4_pred)
bkg4_h_true = bkg4_jets[bkg4_ind_true]
bkg4_h_pred = bkg4_jets[bkg4_ind_pred]

# Delete the bug where b1,b2 indices are the same
bkg1_h_true, bkg1_h_pred = delete_bug(bkg1_h_true, bkg1_h_pred, bkg1_pred, "h")
bkg2_h_true, bkg2_h_pred = delete_bug(bkg2_h_true, bkg2_h_pred, bkg2_pred, "h")
bkg3_h_true, bkg3_h_pred = delete_bug(bkg3_h_true, bkg3_h_pred, bkg3_pred, "h")
bkg4_h_true, bkg4_h_pred = delete_bug(bkg4_h_true, bkg4_h_pred, bkg4_pred, "h")

bkg1_m = mass_h(bkg1_h_pred)
bkg2_m = mass_h(bkg2_h_pred)
bkg3_m = mass_h(bkg3_h_pred)
bkg4_m = mass_h(bkg4_h_pred)
bkg_m = np.hstack((bkg1_m, bkg2_m, bkg3_m, bkg4_m))

plt.figure() 
plt.hist(bkg_m, bins=100, histtype='step', label="predicted background", range=(0,300))
plt.xlim(0,300)
plt.xlabel(r"$m_{bb}$")
plt.legend()
plt.title("TTbb background", pad=15)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"TTbb_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

plt.figure() 
plt.hist(bkg1_m, bins=100, density=True, histtype='step', range=(0,300),
         label="TTbbSemiLeptonic_Powheg_2016_PostVFP")
plt.hist(bkg2_m, bins=100, density=True, histtype='step', range=(0,300),
         label="TTbbSemiLeptonic_Powheg_2016_PreVFP")
plt.hist(bkg3_m, bins=100, density=True, histtype='step', range=(0,300),
         label="TTbbSemiLeptonic_Powheg_2017")
plt.hist(bkg4_m, bins=100, density=True, histtype='step', range=(0,300),
         label="TTbbSemiLeptonic_Powheg_2018")
plt.xlim(0,300)
plt.xlabel(r"$m_{bb}$")
plt.legend(bbox_to_anchor=(1,1))
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"norm_TTbb_split_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ SIGNAL + TTbb ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.figure() 
plt.hist(sig_m, bins=100, density=True, histtype='step', label="inclusive signal", range=(0,300))
plt.hist(sig_matched_m, bins=100, density=True, histtype='step', label="fully matched signal", range=(0,300))
plt.hist(bkg_m, bins=100, density=True, histtype='step', label="TTbb background", range=(0,300))
plt.xlim(0,300)
plt.xlabel(r"$m_{bb}$")
plt.legend()
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"sig_TTbb_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ TTTo BACKGROUND ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nAnalysing TTTo background...\n")

bkg5 = "TTToSemiLeptonic_2016_PostVFP_v9"
bkg5_true, bkg5_pred, bkg5_jets, _, _ = initialise_sig(bkg5)
print("loaded bkg5")
bkg6 = "TTToSemiLeptonic_2016_PreVFP_v9"
bkg6_true, bkg6_pred, bkg6_jets, _, _ = initialise_sig(bkg6)
bkg7 = "TTToSemiLeptonic_2017_v9"
bkg7_true, bkg7_pred, bkg7_jets, _, _ = initialise_sig(bkg7)
#bkg8 = "TTToSemiLeptonic_2018_v9"
#bkg8_true, bkg8_pred, bkg8_jets, _, _ = initialise_sig(bkg8)

"""
basedir = "/eos/user/y/ymaidann/eth_project/Spanet_project/v2_sig_forTrainingDataset/"

bkg7_pred = h5py.File(f'{basedir}/predictions/TTToSemiLeptonic_2017_v0.h5','r')
bkg7_true = h5py.File(f'{basedir}/data/TTToSemiLeptonic_2017.h5','r')

bkg7_jets = ak.from_parquet(f"{basedir}/jets/all_jets_fullRun2_TTToSemiLeptonic_2017_v3.parquet")
(bkg7_jets,_,_,_,_,_,_,_) = ak.unzip(bkg7_jets)
bkg7_jets = ak.with_name(bkg8_jets, name="Momentum4D")
bkg7_jets = bkg7_jets[0:500000]

basedir = "/eos/user/y/ymaidann/eth_project/Spanet_project/v2_sig_bkg"

bkg8_pred = h5py.File(f'{basedir}/predictions/TTToSemiLeptonic_2018_v7.h5','r')
bkg8_true = h5py.File(f'{basedir}/data/TTToSemiLeptonic_2018.h5','r')

bkg8_jets = ak.from_parquet(f"{basedir}/jets/all_jets_fullRun2_TTToSemiLeptonic_2018_v2.parquet")
(bkg8_jets,_,_,_,_,_,_,_) = ak.unzip(bkg8_jets)
bkg8_jets = ak.with_name(bkg8_jets, name="Momentum4D")
bkg8_jets = bkg8_jets[0:1000000]

bkg9_pred = h5py.File(f'{basedir}/predictions/TTToSemiLeptonic_2018_2_v7.h5','r')
bkg9_true = h5py.File(f'{basedir}/data/TTToSemiLeptonic_2018_2.h5','r')

bkg9_jets = ak.from_parquet(f"{basedir}/jets/all_jets_fullRun2_TTToSemiLeptonic_2018_v2.parquet")
(bkg9_jets,_,_,_,_,_,_,_) = ak.unzip(bkg9_jets)
bkg9_jets = ak.with_name(bkg9_jets, name="Momentum4D")
bkg9_jets = bkg9_jets[1000000:2000000]
"""
# Get jets coming from the Higgs
bkg5_ind_true, bkg5_ind_pred = get_higgs_indices(bkg5_true, bkg5_pred)
bkg5_h_true = bkg5_jets[bkg5_ind_true]
bkg5_h_pred = bkg5_jets[bkg5_ind_pred]
bkg6_ind_true, bkg6_ind_pred = get_higgs_indices(bkg6_true, bkg6_pred)
bkg6_h_true = bkg6_jets[bkg6_ind_true]
bkg6_h_pred = bkg6_jets[bkg6_ind_pred]
bkg7_ind_true, bkg7_ind_pred = get_higgs_indices(bkg7_true, bkg7_pred)
bkg7_h_true = bkg7_jets[bkg7_ind_true]
bkg7_h_pred = bkg7_jets[bkg7_ind_pred]
#bkg8_ind_true, bkg8_ind_pred = get_higgs_indices(bkg8_true, bkg8_pred)
#bkg8_h_true = bkg8_jets[bkg8_ind_true]
#bkg8_h_pred = bkg8_jets[bkg8_ind_pred]
"""
bkg9_ind_true, bkg9_ind_pred = get_higgs_indices(bkg9_true, bkg9_pred)
bkg9_h_true = bkg9_jets[bkg9_ind_true]
bkg9_h_pred = bkg9_jets[bkg9_ind_pred]
"""

# Delete predictions where b1 == b2
bkg5_h_true, bkg5_h_pred = delete_bug(bkg5_h_true, bkg5_h_pred, bkg5_pred, "h")
bkg6_h_true, bkg6_h_pred = delete_bug(bkg6_h_true, bkg6_h_pred, bkg6_pred, "h")
bkg7_h_true, bkg7_h_pred = delete_bug(bkg7_h_true, bkg7_h_pred, bkg7_pred, "h")
#bkg8_h_true, bkg8_h_pred = delete_bug(bkg8_h_true, bkg8_h_pred, bkg8_pred, "h")
#bkg9_h_true, bkg9_h_pred = delete_bug(bkg9_h_true, bkg9_h_pred, bkg9_pred, "h")

print("deleted the bug")
bkg5_m = mass_h(bkg5_h_pred)
bkg6_m = mass_h(bkg6_h_pred)
bkg7_m = mass_h(bkg7_h_pred)
#bkg8_m = mass_h(bkg8_h_pred)
#bkg9_m = mass_h(bkg9_h_pred)
#ttt_bkg_m = np.hstack((bkg5_m, bkg6_m, bkg7_m, bkg8_m, bkg9_m))

ttt_bkg_m = np.hstack((bkg5_m, bkg6_m, bkg7_m))

print("Stacked all the masses")

plt.figure() 
plt.hist(ttt_bkg_m, bins=100, histtype='step', label="background", range=(0,300))
plt.xlim(0,300)
plt.xlabel(r"$m_{bb}$")
plt.legend()
plt.title("TTTo background", pad=15)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"TTTo_bkg_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

plt.figure() 
plt.hist(bkg5_m, bins=100, density=True, histtype='step', range=(0,300), label="TTToSemiLeptonic_2016_PostVFP")
plt.hist(bkg6_m, bins=100, density=True, histtype='step', range=(0,300), label="TTToSemiLeptonic_2016_PreVFP")
plt.hist(bkg7_m, bins=100, density=True, histtype='step', range=(0,300), label="TTToSemiLeptonic_2017")
#bkg8_m = np.hstack((bkg8_m, bkg9_m))
#plt.hist(bkg8_m, bins=100, density=True, histtype='step', range=(0,300), label="TTToSemiLeptonic_2018")
plt.xlim(0,300)
plt.xlabel(r"$m_{bb}$")
plt.legend(bbox_to_anchor=(1,1))
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"norm_TTTo_split_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ SIGNAL + TTTo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.figure() 
plt.hist(sig_m, bins=100, density=True, histtype='step', label="inclusive signal", range=(0,300))
plt.hist(sig_matched_m, bins=100, density=True, histtype='step', label="fully matched signal", range=(0,300))
plt.hist(bkg_m, bins=100, density=True, histtype='step', label="TTbb background", range=(0,300))
plt.hist(ttt_bkg_m, bins=100, density=True, histtype='step', label="TTTo background", range=(0,300))
plt.xlim(0,300)
plt.xlabel(r"$m_{bb}$")
plt.legend()
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"sig_TTbb_TTTo_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

plt.figure() 
plt.hist(sig_m, bins=100, density=True, histtype='step', label="inclusive signal", range=(0,300))
plt.hist(sig_matched_m, bins=100, density=True, histtype='step', label="fully matched signal", range=(0,300))
plt.hist(ttt_bkg_m, bins=100, density=True, histtype='step', label="TTTo background", range=(0,300))
plt.xlim(0,300)
plt.xlabel(r"$m_{bb}$")
plt.legend()
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"sig_TTTo_v9.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")
