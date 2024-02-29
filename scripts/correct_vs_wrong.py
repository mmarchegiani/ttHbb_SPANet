# Creates plots of jet pt based on if the jet was predicted correctly or wrongly.
# 2023-10-26 22:35:09
# Yuliia Maidannyk yuliia.maidannyk@ethz.ch

import h5py
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
mpl.rcParams['font.size'] = 26
mpl.rcParams['lines.linewidth'] = 1.5

# Directory to save the plots
figdir = "/eos/user/y/ymaidann/eth_project/Spanet_project/plots/"

# Load files (true .h5, prediction .h5, and jets .parquet)
df_pred = h5py.File('../predictions/0107_output_v7_matched.h5','r')
df_true = h5py.File('../data/tth_matched_3.h5','r')
df_jets = ak.from_parquet("/eos/user/d/dvalsecc/www/ttHbbAnalysis/training_dataset/all_jets_v6.parquet")
(jets,_,_,_,_,lep,met) = ak.unzip(df_jets)
jets = ak.with_name(jets, name="Momentum4D")
lep = ak.with_name(lep, name="Momentum4D")
met = ak.with_name(met, name="Momentum4D")

# Get fully matched jets
higgs = jets[jets.prov == 1]
mask_match = ak.num(higgs) == 2
w_or_t_jets = jets[(jets.prov == 5)|(jets.prov == 2)]
mask_match = mask_match & (ak.num(w_or_t_jets) == 3)
lep_top = jets[jets.prov == 3]
mask_match = mask_match & (ak.num(lep_top) == 1)
jets = jets[mask_match]
lep = lep[mask_match]
met = met[mask_match]

# ~~~~~~~~~~~~~~~~~~~~~~~~ HADRONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n--Hadronic top--\n")

hadtop_index_true, hadtop_index_pred = get_hadtop_indices(df_true, df_pred)
hadtop_jet_pred = jets[hadtop_index_pred]
hadtop_jet_true = jets[hadtop_index_true]

mask = (hadtop_jet_true == hadtop_jet_pred)
hadtop_pred_correct = hadtop_jet_pred[mask] # correctly predicted jets
hadtop_pred_wrong = hadtop_jet_pred[~mask] # wrongly predicted jets

hadtop_pred_correct = hadtop_pred_correct[ak.num(hadtop_pred_correct)==3] # q1,q2,b correct
hadtop_pred_wrong = hadtop_pred_wrong[ak.num(hadtop_pred_wrong)>0]  # drop empty arrays

hadtop_pred_wrong_3 = hadtop_pred_wrong[ak.num(hadtop_pred_wrong)==3] # q1,q2,b wrong
hadtop_pred_wrong_2 = hadtop_pred_wrong[ak.num(hadtop_pred_wrong)==2] # two of the partons are wrong
hadtop_pred_wrong_1 = hadtop_pred_wrong[ak.num(hadtop_pred_wrong)==1] # one of the partons is wrong

hadtop_true_correct = hadtop_jet_true[mask] 
hadtop_true_wrong = hadtop_jet_true[~mask] 

hadtop_true_correct = hadtop_true_correct[ak.num(hadtop_true_correct)==3]
hadtop_true_wrong = hadtop_true_wrong[ak.num(hadtop_true_wrong)>0]

hadtop_true_wrong_3 = hadtop_true_wrong[ak.num(hadtop_true_wrong)==3]
hadtop_true_wrong_2 = hadtop_true_wrong[ak.num(hadtop_true_wrong)==2]
hadtop_true_wrong_1 = hadtop_true_wrong[ak.num(hadtop_true_wrong)==1]

print(f"Number of correctly predicted is {len(hadtop_pred_correct)}")
print(f"Number of wrongly predicted is {len(hadtop_pred_wrong)}")

print(f"Number of 3 jets wrongly predicted is {len(hadtop_pred_wrong_3)}")
print(f"Number of 2 jets wrongly predicted is {len(hadtop_pred_wrong_2)}")
print(f"Number of 1 jet wrongly predicted is {len(hadtop_pred_wrong_1)}")

# Normalised
fig, ax = plt.subplots(1, 1)
plt.hist(pt_t1(hadtop_pred_correct), bins=100, histtype="step", density=True, color='g')
plt.hist(pt_t1(hadtop_pred_wrong_3), bins=100, histtype="step", density=True, color='r')
plt.hist((hadtop_pred_wrong_2[:,0] + hadtop_pred_wrong_2[:,1]).pt,
         bins=100, histtype="step", density=True, color='purple')
plt.hist((hadtop_pred_wrong_1[:,0]).pt, bins=100, histtype="step", density=True, color='b')

plt.legend(labels=["3 jets correct", "3 jets wrong", "2 jets wrong", "1 jet wrong"])
plt.title("Hadronic top (normalised)", pad=15)
plt.xlabel("predicted pt", labelpad=10)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"t1_corr_vs_wrong_pt_v7.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HIGGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n--Higgs--\n")

higgs_index_true, higgs_index_pred = get_higgs_indices(df_true, df_pred)
higgs_jet_pred = jets[higgs_index_pred]
higgs_jet_true = jets[higgs_index_true]

mask = (higgs_jet_true == higgs_jet_pred)
higgs_pred_correct = higgs_jet_pred[mask] 
higgs_pred_wrong = higgs_jet_pred[~mask]

higgs_pred_correct = higgs_pred_correct[ak.num(higgs_pred_correct)==2]
higgs_pred_wrong = higgs_pred_wrong[ak.num(higgs_pred_wrong)>0]

higgs_pred_wrong_2 = higgs_pred_wrong[ak.num(higgs_pred_wrong)==2]
higgs_pred_wrong_1 = higgs_pred_wrong[ak.num(higgs_pred_wrong)==1]

mask = (higgs_jet_true == higgs_jet_pred)
higgs_true_correct = higgs_jet_true[mask] # correctly predicted jets 
higgs_true_wrong = higgs_jet_true[~mask] # wrongly predicted jets

higgs_true_correct = higgs_true_correct[ak.num(higgs_true_correct)==2]
higgs_true_wrong = higgs_true_wrong[ak.num(higgs_true_wrong)>0]

higgs_true_wrong_2 = higgs_true_wrong[ak.num(higgs_true_wrong)==2]
higgs_true_wrong_1 = higgs_true_wrong[ak.num(higgs_true_wrong)==1]

print(f"Number of correctly predicted is {len(higgs_pred_correct)}")
print(f"Number of wrongly predicted is {len(higgs_pred_wrong)}")

print(f"Number of 2 jets wrongly predicted is {len(higgs_pred_wrong_2)}")
print(f"Number of 1 jet wrongly predicted is {len(higgs_pred_wrong_1)}")

# Normalised 
fig, ax = plt.subplots(1, 1)
plt.hist(pt_h(higgs_pred_correct), bins=100, histtype="step", density=True, range=(0,800), color='g')
plt.hist(pt_h(higgs_pred_wrong_2), bins=100, histtype="step", density=True, range=(0,800), color='r')
plt.hist((higgs_pred_wrong_1[:,0]).pt, bins=100, histtype="step",
         density=True, range=(0,800), color='purple')

plt.legend(labels=["2 jets correct", "2 jets wrong", "1 jet wrong"])
plt.title("Higgs (normalised)", pad=15)
plt.xlabel("predicted pt", labelpad=10)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"h_corr_vs_wrong_pt_v7.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ LEPTONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n--Leptonic top--\n")

leptop_index_true, leptop_index_pred = get_leptop_indices(df_true, df_pred)
leptop_jet_pred = jets[leptop_index_pred]
leptop_jet_true = jets[leptop_index_true]

mask = (leptop_jet_true == leptop_jet_pred)
leptop_pred_correct = leptop_jet_pred[mask] # correctly predicted jets
leptop_pred_wrong = leptop_jet_pred[~mask] # wrongly predicted jets

# Drop empty arrays
leptop_pred_correct = leptop_pred_correct[ak.num(leptop_pred_correct)>0]
leptop_pred_wrong = leptop_pred_wrong[ak.num(leptop_pred_wrong)>0]

print(f"Number of correctly predicted is {len(leptop_pred_correct)}")
print(f"Number of wrongly predicted is {len(leptop_pred_wrong)}")

# Normalised
fig, ax = plt.subplots(1, 1)
plt.hist(leptop_pred_correct[:,0].pt, density=True, bins=100, color='g', histtype="step")
plt.hist(leptop_pred_wrong[:,0].pt, density=True, bins=100, color='r', histtype="step")
plt.legend(labels=["correct", "wrong"])
plt.title("Leptonic top (normalised)", pad=15)
plt.xlabel("b predicted pt", labelpad=10)
plt.rcParams['figure.facecolor'] = 'white'
name = figdir+"t2_corr_vs_wrong_pt_v7.png"
plt.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")
