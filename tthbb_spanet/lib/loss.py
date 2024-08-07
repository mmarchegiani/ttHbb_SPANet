import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ROOT)

def parse_tensorboard(ea, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

def filter_scalars(scalars, key):
    if type(key) == str:
        return list(filter(lambda x : key in x, scalars))
    elif type(key) == list:
        total_list = []
        for k in key:
            total_list += list(filter(lambda x : k in x, scalars))
        return total_list

def get_loss_by_epoch(loss):
    epoch = loss["epoch"].values
    total_loss = loss["loss/total_loss"]
    total_loss["epoch"] = epoch.value.astype(int)
    total_loss_by_epoch = total_loss[["epoch", "value"]].groupby("epoch").mean()
    return total_loss_by_epoch

def plot_loss(model_folder, save=True):
    ls = os.listdir(model_folder)
    filename = list(filter(lambda x : "tfevents" in x, ls))
    if len(filename) == 0:
        sys.exit("No tfevents file found in the model folder")
    elif len(filename) > 1:
        sys.exit("Multiple tfevents file found in the model folder")
    else:
        filename = os.path.join(model_folder, filename[0])
    print("Reading file: ", filename)
    ea = event_accumulator.EventAccumulator(
        filename,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    scalars = ea.Tags()["scalars"]
    sc = filter_scalars(scalars, ["loss", "epoch"])
    df = parse_tensorboard(ea, sc)
    epoch = df["epoch"]
    total_loss = df["loss/total_loss"]
    total_loss["epoch"] = epoch.value.astype(int)
    total_loss_by_epoch = get_loss_by_epoch(total_loss)
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(total_loss_by_epoch, label="Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.legend(loc="upper right")
    ax.savefig(os.path.join(model_folder, "loss.png"), dpi=300)
