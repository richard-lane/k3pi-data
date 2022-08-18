"""
Find the absolute efficiency for particle gun events

Choose to find these from particle gun, as we have the truth information
readily available

"""
import sys
import pathlib
import argparse
from typing import Tuple
from multiprocessing import Manager, Process
import numpy as np
import matplotlib.pyplot as plt
import uproot

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import definitions
from lib_data import cuts


def _n_reco(data_tree, hlt_tree) -> int:
    """
    Number of events reconstructed

    """
    # Mask to perform straight cuts
    keep = cuts.sanity_keep(data_tree)

    # Combine with the mask to perform HLT cuts
    keep &= cuts.hlt_keep_pgun(hlt_tree)

    # Combine with the background category mask
    keep &= cuts.bkgcat(data_tree)

    return np.sum(keep)


def _efficiency_and_err(n_gen: int, n_reco: int) -> Tuple[float, float]:
    """
    Efficiency and its error (assuming Poisson errors) in %

    """
    efficiency = 100 * n_reco / n_gen

    err = efficiency * np.sqrt((1 / n_reco) + (1 / n_gen))

    return efficiency, err


def _calculate(source_dir, label, out_dict) -> None:
    """
    Calculate efficiency and error for the files in the source dir
    Populate out_dict with label + "_eff" and "label" + "_err"

    """
    # Some folders might break
    broken_folders = []

    n_reco, n_gen = 0, 0

    # Iterate over files, incrementing counters
    for i, folder in enumerate(source_dir.glob("*")):
        if not i % 10:
            print(f"{label} {i}")
        try:
            with uproot.open(folder / "pGun_TRACK.root") as data_f, uproot.open(
                folder / "Hlt1TrackMVA.root"
            ) as hlt_f:
                reco_tree = data_f["Dstp2D0pi/DecayTree"]
                gen_tree = data_f["MCDstp2D0pi/MCDecayTree"]
                hlt_tree = hlt_f["DecayTree"]

                # Do cuts on the reco bit, count the entries
                n_reco += _n_reco(reco_tree, hlt_tree)

                # No cuts for the generator level
                n_gen += gen_tree.numentries

        except FileNotFoundError:
            broken_folders.append(str(folder))
            continue

    if broken_folders:
        print(f"{label}: Failed to read from dirs:\n\t{broken_folders}")
        print(
            "This may be expected, e.g. there may be already merged files also in the dir"
        )

    efficiency, err = _efficiency_and_err(n_gen, n_reco)
    out_dict[f"{label}_eff"] = efficiency
    out_dict[f"{label}_err"] = err


def _plot(labels: Tuple, efficiencies: Tuple, errors: Tuple) -> None:
    """
    Plot efficiencies

    """
    fig, ax = plt.subplots()
    for i, (label, efficiency, error) in enumerate(zip(labels, efficiencies, errors)):
        ax.errorbar(
            [i],
            efficiency,
            yerr=error,
            label=label,
            fmt="k+",
        )

    ax.set_xticks([i for i, _ in enumerate(labels)])
    ax.set_xticklabels(labels)

    fig.savefig("efficiencies.png")


def main():
    """
    Open the folder containing the particle gun files that we want
    Open the MC truth and the reconstructed events in each folder
    Apply cuts to the reconstructed events
    Count how many of each type there are
    Report this by just printing it out, for now

    """
    out_dict = Manager().dict()
    labels = ("false", "cf", "dcs")
    procs = [
        Process(target=_calculate, args=(source_dir, label, out_dict))
        for source_dir, label in zip(
            (
                definitions.FALSE_SIGN_SOURCE_DIR,
                definitions.RS_PGUN_SOURCE_DIR,
                definitions.WS_PGUN_SOURCE_DIR,
            ),
            labels,
        )
    ]

    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    _plot(
        labels,
        [out_dict[f"{label}_eff"] for label in labels],
        [out_dict[f"{label}_err"] for label in labels],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the average efficiency for particle gun events"
        "Runs parallel processes for false sign, CF and DCS."
    )
    args = parser.parse_args()

    main()
