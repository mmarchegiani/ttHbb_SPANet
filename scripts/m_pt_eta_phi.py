# Generates plots of kinematic variables (invariant mass, pt, eta, phi) for ttH
# 2023-10-26 22:31:32
# Yuliia Maidannyk yuliia.maidannyk@ethz.ch

import awkward as ak
import vector
import h5py
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

fig, axs = plt.subplots(2, 2)
hadtop_index_true, hadtop_index_pred = get_hadtop_indices(df_true, df_pred)
hadtop_jet_pred = jets[hadtop_index_pred]
hadtop_jet_true = jets[hadtop_index_true]

axs[0,0].hist(mass_t1(hadtop_jet_pred), bins=100, range=(0,350), histtype="step", density=True, lw=1.5)
axs[0,1].hist(pt_t1(hadtop_jet_pred), bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[1,0].hist(eta_t1(hadtop_jet_pred), bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)
axs[1,1].hist(phi_t1(hadtop_jet_pred), bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)

axs[0,0].hist(mass_t1(hadtop_jet_true), bins=100, range=(0,350), histtype="step", density=True, lw=1.5)
axs[0,1].hist(pt_t1(hadtop_jet_true), bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[1,0].hist(eta_t1(hadtop_jet_true), bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)
axs[1,1].hist(phi_t1(hadtop_jet_true), bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)

axs[0,0].set_title('mass', fontsize=18)
axs[0,1].set_title('pt', fontsize=18)
axs[1,0].set_title('eta', fontsize=18)
axs[1,1].set_title('phi', fontsize=18)
axs[0,0].tick_params(labelsize=10)
axs[0,1].tick_params(labelsize=10)
axs[1,0].tick_params(labelsize=10)
axs[1,1].tick_params(labelsize=10)

labels = ["predicted", "true"]
fig.legend(labels, loc="upper center", ncol=2)
name = figdir+"t1_m_pt_eta_phi_v7.png"
fig.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HIGGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig, axs = plt.subplots(2, 2)
higgs_index_true, higgs_index_pred = get_higgs_indices(df_true, df_pred)
higgs_jet_pred = jets[higgs_index_pred]
higgs_jet_true = jets[higgs_index_true]

axs[0,0].hist(mass_h(higgs_jet_pred), bins=100, range=(0,350), histtype="step", density=True, lw=1.5)
axs[0,1].hist(pt_h(higgs_jet_pred), bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[1,0].hist(eta_h(higgs_jet_pred), bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)
axs[1,1].hist(phi_h(higgs_jet_pred), bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)

axs[0,0].hist(mass_h(higgs_jet_true), bins=100, range=(0,350), histtype="step", density=True, lw=1.5)
axs[0,1].hist(pt_h(higgs_jet_true), bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[1,0].hist(eta_h(higgs_jet_true), bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)
axs[1,1].hist(phi_h(higgs_jet_true), bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)

axs[0,0].set_title('mass', fontsize=18)
axs[0,1].set_title('pt', fontsize=18)
axs[1,0].set_title('eta', fontsize=18)
axs[1,1].set_title('phi', fontsize=18)
axs[0,0].tick_params(labelsize=10)
axs[0,1].tick_params(labelsize=10)
axs[1,0].tick_params(labelsize=10)
axs[1,1].tick_params(labelsize=10)

labels = ["predicted", "true"]
fig.legend(labels, loc="upper center", ncol=2)
name = figdir+"h_m_pt_eta_phi_v7.png"
fig.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ LEPTONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig, axs = plt.subplots(2, 2)
leptop_index_true, leptop_index_pred = get_leptop_indices(df_true, df_pred)
leptop_jet_pred = jets[leptop_index_pred]
leptop_jet_true = jets[leptop_index_true]

axs[0,0].hist((lep + met + leptop_jet_pred[:,0]).m, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[0,1].hist((lep + met + leptop_jet_pred[:,0]).pt, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[1,0].hist((lep + met + leptop_jet_pred[:,0]).eta, bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)
axs[1,1].hist((lep + met + leptop_jet_pred[:,0]).phi, bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)

axs[0,0].hist((lep + met + leptop_jet_true[:,0]).m, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[0,1].hist((lep + met + leptop_jet_true[:,0]).pt, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[1,0].hist((lep + met + leptop_jet_true[:,0]).eta, bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)
axs[1,1].hist((lep + met + leptop_jet_true[:,0]).phi, bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)

axs[0,0].set_title('mass', fontsize=18)
axs[0,1].set_title('pt', fontsize=18)
axs[1,0].set_title('eta', fontsize=18)
axs[1,1].set_title('phi', fontsize=18)
axs[0,0].tick_params(labelsize=10)
axs[0,1].tick_params(labelsize=10)
axs[1,0].tick_params(labelsize=10)
axs[1,1].tick_params(labelsize=10)

labels = ["predicted", "true"]
fig.legend(labels, loc="upper center", ncol=2)
name = figdir+"t2_m_pt_eta_phi_v7.png"
fig.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")