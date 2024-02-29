# Generates plots of kinematic variables (invariant mass, pt, eta, phi) for ttH background files.
# 2023-10-26 22:26:56
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

# Directory to save the plots
figdir = "/eos/user/y/ymaidann/eth_project/Spanet_project/plots/"

# ~~~~~~~~~~~~~~~~~~~~~~~~ HADRONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nAnalysing hadronic top...\n")
fig, axs = plt.subplots(2, 2)

# Load files (true .h5, prediction .h5, and jets .parquet)
bkg1 = "TTbbSemiLeptonic_Powheg_2016_PostVFP_v9"
bkg1_true, bkg1_pred, bkg1_jets, bkg_lep1, bkg_met1 = initialise_sig(bkg1)
bkg2 = "TTbbSemiLeptonic_Powheg_2016_PreVFP_v9"
bkg2_true, bkg2_pred, bkg2_jets, bkg_lep2, bkg_met2 = initialise_sig(bkg2)
bkg3 = "TTbbSemiLeptonic_Powheg_2017_v9"
bkg3_true, bkg3_pred, bkg3_jets, bkg_lep3, bkg_met3 = initialise_sig(bkg3)
bkg4 = "TTbbSemiLeptonic_Powheg_2018_v9"
bkg4_true, bkg4_pred, bkg4_jets, bkg_lep4, bkg_met4 = initialise_sig(bkg4)

ind_true, ind_pred = get_hadtop_indices(bkg1_true, bkg1_pred)
jet_pred1 = bkg1_jets[ind_pred]
jet_true1 = bkg1_jets[ind_true]
ind_true, ind_pred = get_hadtop_indices(bkg2_true, bkg2_pred)
jet_pred2 = bkg2_jets[ind_pred]
jet_true2 = bkg2_jets[ind_true]
ind_true, ind_pred = get_hadtop_indices(bkg3_true, bkg3_pred)
jet_pred3 = bkg3_jets[ind_pred]
jet_true3 = bkg3_jets[ind_true]
ind_true, ind_pred = get_hadtop_indices(bkg4_true, bkg4_pred)
jet_pred4 = bkg4_jets[ind_pred]
jet_true4 = bkg4_jets[ind_true]

# Delete the bug where q1==q2
jet_true1, jet_pred1 = delete_bug(jet_true1, jet_pred1, bkg1_pred, "t1")
jet_true2, jet_pred2 = delete_bug(jet_true2, jet_pred2, bkg2_pred, "t1")
jet_true3, jet_pred3 = delete_bug(jet_true3, jet_pred3, bkg3_pred, "t1")
jet_true4, jet_pred4 = delete_bug(jet_true4, jet_pred4, bkg4_pred, "t1")

# Get T1 matched events only (otherwise the idea of mass doesn't make sense)
jet_true1, jet_pred1 = event_particle_matched(jet_true1, jet_pred1)
jet_true2, jet_pred2 = event_particle_matched(jet_true2, jet_pred2)
jet_true3, jet_pred3 = event_particle_matched(jet_true3, jet_pred3)
jet_true4, jet_pred4 = event_particle_matched(jet_true4, jet_pred4)
    
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

axs[0,0].hist(pred, bins=50, range=(0,350), histtype="step", density=True, lw=1.5)
axs[0,0].hist(true, bins=50, range=(0,300), histtype="step", density=True, lw=1.5)

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

name = figdir+"t1_m_pt_eta_phi_bkg_v9.png"
fig.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")

# ~~~~~~~~~~~~~~~~~~~~~~~~ LEPTONIC TOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nAnalysing leptonic top...\n")
fig, axs = plt.subplots(2, 2)

ind_true, ind_pred = get_leptop_indices(bkg1_true, bkg1_pred)
jet_pred1 = bkg1_jets[ind_pred]
jet_true1 = bkg1_jets[ind_true]

ind_true, ind_pred = get_leptop_indices(bkg2_true, bkg2_pred)
jet_pred2 = bkg2_jets[ind_pred]
jet_true2 = bkg2_jets[ind_true]

ind_true, ind_pred = get_leptop_indices(bkg3_true, bkg3_pred)
jet_pred3 = bkg3_jets[ind_pred]
jet_true3 = bkg3_jets[ind_true]

ind_true, ind_pred = get_leptop_indices(bkg4_true, bkg4_pred)
jet_pred4 = bkg4_jets[ind_pred]
jet_true4 = bkg4_jets[ind_true]

# Get T2 matched events only (otherwise the idea of mass doesn't make sense)
jet_true1, jet_pred1, bkg_lep1, bkg_met1 = event_particle_matched(jet_true1, jet_pred1, bkg_lep1, bkg_met1)
jet_true2, jet_pred2, bkg_lep2, bkg_met2 = event_particle_matched(jet_true2, jet_pred2, bkg_lep2, bkg_met2)
jet_true3, jet_pred3, bkg_lep3, bkg_met3 = event_particle_matched(jet_true3, jet_pred3, bkg_lep3, bkg_met3)
jet_true4, jet_pred4, bkg_lep4, bkg_met4 = event_particle_matched(jet_true4, jet_pred4, bkg_lep4, bkg_met4)

pred1 = mass_t2(jet_pred1, bkg_lep1, bkg_met1)
pred2 = mass_t2(jet_pred2, bkg_lep2, bkg_met2)
pred3 = mass_t2(jet_pred3, bkg_lep3, bkg_met3)
pred4 = mass_t2(jet_pred4, bkg_lep4, bkg_met4)
true1 = mass_t2(jet_true1, bkg_lep1, bkg_met1)
true2 = mass_t2(jet_true2, bkg_lep2, bkg_met2)
true3 = mass_t2(jet_true3, bkg_lep3, bkg_met3)
true4 = mass_t2(jet_true4, bkg_lep4, bkg_met4)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[0,0].hist(pred, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[0,0].hist(true, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)

pred1 = pt_t2(jet_pred1, bkg_lep1, bkg_met1)
pred2 = pt_t2(jet_pred2, bkg_lep2, bkg_met2)
pred3 = pt_t2(jet_pred3, bkg_lep3, bkg_met3)
pred4 = pt_t2(jet_pred4, bkg_lep4, bkg_met4)
true1 = pt_t2(jet_true1, bkg_lep1, bkg_met1)
true2 = pt_t2(jet_true2, bkg_lep2, bkg_met2)
true3 = pt_t2(jet_true3, bkg_lep3, bkg_met3)
true4 = pt_t2(jet_true4, bkg_lep4, bkg_met4)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[0,1].hist(pred, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)
axs[0,1].hist(true, bins=100, range=(0,800), histtype="step", density=True, lw=1.5)

pred1 = eta_t2(jet_pred1, bkg_lep1, bkg_met1)
pred2 = eta_t2(jet_pred2, bkg_lep2, bkg_met2)
pred3 = eta_t2(jet_pred3, bkg_lep3, bkg_met3)
pred4 = eta_t2(jet_pred4, bkg_lep4, bkg_met4)
true1 = eta_t2(jet_true1, bkg_lep1, bkg_met1)
true2 = eta_t2(jet_true2, bkg_lep2, bkg_met2)
true3 = eta_t2(jet_true3, bkg_lep3, bkg_met3)
true4 = eta_t2(jet_true4, bkg_lep4, bkg_met4)

pred = np.hstack((pred1, pred2, pred3, pred4))
true = np.hstack((true1, true2, true3, true4))

axs[1,0].hist(pred, bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)
axs[1,0].hist(true, bins=100, range=(-5,5), histtype="step", density=True, lw=1.5)

pred1 = phi_t2(jet_pred1, bkg_lep1, bkg_met1)
pred2 = phi_t2(jet_pred2, bkg_lep2, bkg_met2)
pred3 = phi_t2(jet_pred3, bkg_lep3, bkg_met3)
pred4 = phi_t2(jet_pred4, bkg_lep4, bkg_met4)
true1 = phi_t2(jet_true1, bkg_lep1, bkg_met1)
true2 = phi_t2(jet_true2, bkg_lep2, bkg_met2)
true3 = phi_t2(jet_true3, bkg_lep3, bkg_met3)
true4 = phi_t2(jet_true4, bkg_lep4, bkg_met4)

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

name = figdir+"t2_m_pt_eta_phi_bkg_v9.png"
fig.savefig(f"{name}", transparent=False, dpi=300, bbox_inches='tight')
print(f"Figure saved as : {name}")
