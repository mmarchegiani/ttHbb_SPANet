# Generates plots of kinematic variables (invariant mass, pt, eta, phi) for ttH signal files.
# 2023-10-26 22:27:45
# Yuliia Maidannyk yuliia.maidannyk@ethz.ch

import awkward as ak
import pandas as pd
import vector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep
hep.style.use(hep.style.ROOT)
vector.register_awkward()

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
    Get only fully matched jets. Output jets and the fully matched mask
    """
    higgs = jets[jets.prov == 1]
    mask_match = ak.num(higgs) == 2

    w_or_t_jets = jets[(jets.prov == 5)|(jets.prov == 2)]
    mask_match = mask_match & (ak.num(w_or_t_jets) == 3)

    lep_top = jets[jets.prov == 3]
    mask_match = mask_match & (ak.num(lep_top) == 1)

    jets = jets[mask_match]
    return jets, mask_match

# Directory to save the plots
figdir = "/eos/user/y/ymaidann/eth_project/Spanet_project/plots/"

# Load files (true .h5, prediction .h5, and jets .parquet)
sig1 = "ttHTobb_forTraining_2017_matched_v9"
sig1_true, sig1_pred, sig1_jets, sig1_lep, sig1_met = initialise_sig(sig1)
sig2 = "ttHTobb_forTraining_2018_matched_v9"
sig2_true, sig2_pred, sig2_jets, sig2_lep, sig2_met = initialise_sig(sig2)
sig3 = "ttHTobb_forTraining_2016_PreVFP_matched_v9"
sig3_true, sig3_pred, sig3_jets, sig3_lep, sig3_met = initialise_sig(sig3)
sig4 = "ttHTobb_forTraining_2016_PostVFP_matched_v9"
sig4_true, sig4_pred, sig4_jets, sig4_lep, sig4_met = initialise_sig(sig4)

sig1_jets, mask_match = get_matched(sig1_jets)
sig1_lep = sig1_lep[mask_match]
sig1_met = sig1_met[mask_match]
sig2_jets, mask_match = get_matched(sig2_jets)
sig2_lep = sig2_lep[mask_match]
sig2_met = sig2_met[mask_match]
sig3_jets, mask_match = get_matched(sig3_jets)
sig3_lep = sig3_lep[mask_match]
sig3_met = sig3_met[mask_match]
sig4_jets, mask_match = get_matched(sig4_jets)
sig4_lep = sig4_lep[mask_match]
sig4_met = sig4_met[mask_match]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HIGGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nAnalysing Higgs...\n")
fig, axs = plt.subplots(2, 2)

higgs_ind_true, higgs_ind_pred = get_higgs_indices(sig1_true, sig1_pred)
higgs_jet_pred1 = sig1_jets[higgs_ind_pred]
higgs_jet_true1 = sig1_jets[higgs_ind_true]
higgs_ind_true, higgs_ind_pred = get_higgs_indices(sig2_true, sig2_pred)
higgs_jet_pred2 = sig2_jets[higgs_ind_pred]
higgs_jet_true2 = sig2_jets[higgs_ind_true]
higgs_ind_true, higgs_ind_pred = get_higgs_indices(sig3_true, sig3_pred)
higgs_jet_pred3 = sig3_jets[higgs_ind_pred]
higgs_jet_true3 = sig3_jets[higgs_ind_true]
higgs_ind_true, higgs_ind_pred = get_higgs_indices(sig4_true, sig4_pred)
higgs_jet_pred4 = sig4_jets[higgs_ind_pred]
higgs_jet_true4 = sig4_jets[higgs_ind_true]

pred1 = mass_h(higgs_jet_pred1)
pred2 = mass_h(higgs_jet_pred2)
pred3 = mass_h(higgs_jet_pred3)
pred4 = mass_h(higgs_jet_pred4)
true1 = mass_h(higgs_jet_true1)
true2 = mass_h(higgs_jet_true2)
true3 = mass_h(higgs_jet_true3)
true4 = mass_h(higgs_jet_true4)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[0,0].hist(pred, bins=100, range=(0,300), histtype="step", density=True, lw=1.5)
axs[0,0].hist(true, bins=100, range=(0,300), histtype="step", density=True, lw=1.5)

pred1 = pt_h(higgs_jet_pred1)
pred2 = pt_h(higgs_jet_pred2)
pred3 = pt_h(higgs_jet_pred3)
pred4 = pt_h(higgs_jet_pred4)
true1 = pt_h(higgs_jet_true1)
true2 = pt_h(higgs_jet_true2)
true3 = pt_h(higgs_jet_true3)
true4 = pt_h(higgs_jet_true4)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[0,1].hist(pred, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[0,1].hist(true, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)

pred1 = eta_h(higgs_jet_pred1)
pred2 = eta_h(higgs_jet_pred2)
pred3 = eta_h(higgs_jet_pred3)
pred4 = eta_h(higgs_jet_pred4)
true1 = eta_h(higgs_jet_true1)
true2 = eta_h(higgs_jet_true2)
true3 = eta_h(higgs_jet_true3)
true4 = eta_h(higgs_jet_true4)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[1,0].hist(pred, bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)
axs[1,0].hist(true, bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)

pred1 = phi_h(higgs_jet_pred1)
pred2 = phi_h(higgs_jet_pred2)
pred3 = phi_h(higgs_jet_pred3)
pred4 = phi_h(higgs_jet_pred4)
true1 = phi_h(higgs_jet_true1)
true2 = phi_h(higgs_jet_true2)
true3 = phi_h(higgs_jet_true3)
true4 = phi_h(higgs_jet_true4)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[1,1].hist(pred, bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)
axs[1,1].hist(true, bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)

axs[0,0].set_xlabel('Invariant mass [GeV]', fontsize=12, labelpad=10)
axs[0,1].set_xlabel('Transverse momentum [GeV]', fontsize=12, labelpad=10)
axs[1,0].set_xlabel('Pseudorapidity', fontsize=12, labelpad=10)
axs[1,1].set_xlabel('Azimuthal angle [rad]', fontsize=12, labelpad=10)
axs[0,0].tick_params(labelsize=10)
axs[0,1].tick_params(labelsize=10)
axs[1,0].tick_params(labelsize=10)
axs[1,1].tick_params(labelsize=10)
labels = ["predicted", "true"]
fig.legend(labels, loc="upper center", ncol=2)

name = figdir+"h_m_pt_eta_phi_sig_v9.png"
fig.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~ HADRONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nAnalysing hadronic top...\n")
fig, axs = plt.subplots(2, 2)

ind_true, ind_pred = get_hadtop_indices(sig1_true, sig1_pred)
jet_pred1 = sig1_jets[ind_pred]
jet_true1 = sig1_jets[ind_true]
ind_true, ind_pred = get_hadtop_indices(sig2_true, sig2_pred)
jet_pred2 = sig2_jets[ind_pred]
jet_true2 = sig2_jets[ind_true]
ind_true, ind_pred = get_hadtop_indices(sig3_true, sig3_pred)
jet_pred3 = sig3_jets[ind_pred]
jet_true3 = sig3_jets[ind_true]
ind_true, ind_pred = get_hadtop_indices(sig4_true, sig4_pred)
jet_pred4 = sig4_jets[ind_pred]
jet_true4 = sig4_jets[ind_true]

pred1 = mass_t1(jet_pred1)
pred2 = mass_t1(jet_pred2)
pred3 = mass_t1(jet_pred3)
pred4 = mass_t1(jet_pred4)
true1 = mass_t1(jet_true1)
true2 = mass_t1(jet_true2)
true3 = mass_t1(jet_true3)
true4 = mass_t1(jet_true4)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[0,0].hist(pred, bins=100, range=(0,300), histtype="step", density=True, lw=1.5)
axs[0,0].hist(true, bins=100, range=(0,300), histtype="step", density=True, lw=1.5)

pred1 = pt_t1(jet_pred1)
pred2 = pt_t1(jet_pred2)
pred3 = pt_t1(jet_pred3)
pred4 = pt_t1(jet_pred4)
true1 = pt_t1(jet_true1)
true2 = pt_t1(jet_true2)
true3 = pt_t1(jet_true3)
true4 = pt_t1(jet_true4)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[0,1].hist(pred, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[0,1].hist(true, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)

pred1 = eta_t1(jet_pred1)
pred2 = eta_t1(jet_pred2)
pred3 = eta_t1(jet_pred3)
pred4 = eta_t1(jet_pred4)
true1 = eta_t1(jet_true1)
true2 = eta_t1(jet_true2)
true3 = eta_t1(jet_true3)
true4 = eta_t1(jet_true4)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[1,0].hist(pred, bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)
axs[1,0].hist(true, bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)

pred1 = phi_t1(jet_pred1)
pred2 = phi_t1(jet_pred2)
pred3 = phi_t1(jet_pred3)
pred4 = phi_t1(jet_pred4)
true1 = phi_t1(jet_true1)
true2 = phi_t1(jet_true2)
true3 = phi_t1(jet_true3)
true4 = phi_t1(jet_true4)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[1,1].hist(pred, bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)
axs[1,1].hist(true, bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)

axs[0,0].set_xlabel('Invariant mass [GeV]', fontsize=12, labelpad=10)
axs[0,1].set_xlabel('Transverse momentum [GeV]', fontsize=12, labelpad=10)
axs[1,0].set_xlabel('Pseudorapidity', fontsize=12, labelpad=10)
axs[1,1].set_xlabel('Azimuthal angle [rad]', fontsize=12, labelpad=10)
axs[0,0].tick_params(labelsize=10)
axs[0,1].tick_params(labelsize=10)
axs[1,0].tick_params(labelsize=10)
axs[1,1].tick_params(labelsize=10)
labels = ["predicted", "true"]
fig.legend(labels, loc="upper center", ncol=2)

name = figdir+"t1_m_pt_eta_phi_sig_v9.png"
fig.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~ LEPTONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nAnalysing leptonic top...\n")
fig, axs = plt.subplots(2, 2)

ind_true, ind_pred = get_leptop_indices(sig1_true, sig1_pred)
jet_pred1 = sig1_jets[ind_pred]
jet_true1 = sig1_jets[ind_true]
ind_true, ind_pred = get_leptop_indices(sig2_true, sig2_pred)
jet_pred2 = sig2_jets[ind_pred]
jet_true2 = sig2_jets[ind_true]
ind_true, ind_pred = get_leptop_indices(sig3_true, sig3_pred)
jet_pred3 = sig3_jets[ind_pred]
jet_true3 = sig3_jets[ind_true]
ind_true, ind_pred = get_leptop_indices(sig4_true, sig4_pred)
jet_pred4 = sig4_jets[ind_pred]
jet_true4 = sig4_jets[ind_true]

pred1 = mass_t2(jet_pred1, sig1_lep, sig1_met)
pred2 = mass_t2(jet_pred2, sig2_lep, sig2_met)
pred3 = mass_t2(jet_pred3, sig3_lep, sig3_met)
pred4 = mass_t2(jet_pred4, sig4_lep, sig4_met)
true1 = mass_t2(jet_true1, sig1_lep, sig1_met)
true2 = mass_t2(jet_true2, sig2_lep, sig2_met)
true3 = mass_t2(jet_true3, sig3_lep, sig3_met)
true4 = mass_t2(jet_true4, sig4_lep, sig4_met)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[0,0].hist(pred, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[0,0].hist(true, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)

pred1 = pt_t2(jet_pred1, sig1_lep, sig1_met)
pred2 = pt_t2(jet_pred2, sig2_lep, sig2_met)
pred3 = pt_t2(jet_pred3, sig3_lep, sig3_met)
pred4 = pt_t2(jet_pred4, sig4_lep, sig4_met)
true1 = pt_t2(jet_true1, sig1_lep, sig1_met)
true2 = pt_t2(jet_true2, sig2_lep, sig2_met)
true3 = pt_t2(jet_true3, sig3_lep, sig3_met)
true4 = pt_t2(jet_true4, sig4_lep, sig4_met)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[0,1].hist(pred, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[0,1].hist(true, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)

pred1 = eta_t2(jet_pred1, sig1_lep, sig1_met)
pred2 = eta_t2(jet_pred2, sig2_lep, sig2_met)
pred3 = eta_t2(jet_pred3, sig3_lep, sig3_met)
pred4 = eta_t2(jet_pred4, sig4_lep, sig4_met)
true1 = eta_t2(jet_true1, sig1_lep, sig1_met)
true2 = eta_t2(jet_true2, sig2_lep, sig2_met)
true3 = eta_t2(jet_true3, sig3_lep, sig3_met)
true4 = eta_t2(jet_true4, sig4_lep, sig4_met)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[1,0].hist(pred, bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)
axs[1,0].hist(true, bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)

pred1 = phi_t2(jet_pred1, sig1_lep, sig1_met)
pred2 = phi_t2(jet_pred2, sig2_lep, sig2_met)
pred3 = phi_t2(jet_pred3, sig3_lep, sig3_met)
pred4 = phi_t2(jet_pred4, sig4_lep, sig4_met)
true1 = phi_t2(jet_true1, sig1_lep, sig1_met)
true2 = phi_t2(jet_true2, sig2_lep, sig2_met)
true3 = phi_t2(jet_true3, sig3_lep, sig3_met)
true4 = phi_t2(jet_true4, sig4_lep, sig4_met)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[1,1].hist(pred, bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)
axs[1,1].hist(true, bins=50, range=(-np.pi,np.pi), histtype="step", density=True, lw=1.5)

axs[0,0].set_xlabel('Invariant mass [GeV]', fontsize=12, labelpad=10)
axs[0,1].set_xlabel('Transverse momentum [GeV]', fontsize=12, labelpad=10)
axs[1,0].set_xlabel('Pseudorapidity', fontsize=12, labelpad=10)
axs[1,1].set_xlabel('Azimuthal angle [rad]', fontsize=12, labelpad=10)

axs[0,0].tick_params(labelsize=10)
axs[0,1].tick_params(labelsize=10)
axs[1,0].tick_params(labelsize=10)
axs[1,1].tick_params(labelsize=10)
labels = ["predicted", "true"]
fig.legend(labels, loc="upper center", ncol=2)

name = figdir+"t2_m_pt_eta_phi_sig_v9.png"
fig.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")
