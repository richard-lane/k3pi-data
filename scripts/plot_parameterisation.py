"""
Make plots of phase space variables to make sure everything looks like we expect it to

"""
import sys
import pathlib
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from fourbody.param import helicity_param

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import definitions, get, util


def _plot(points: List[np.ndarray], labels: List[str]) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a list of points and labels on an axis; return the figure and axis

    """

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    hist_kw = {"density": True, "histtype": "step"}
    for i, axis in tqdm(enumerate(ax.ravel())):
        for arr, label in zip(points, labels):
            # Might want to plot up to some maximum lifetime, to illustrate something
            max_lifetimes = np.inf
            mask = arr[:, -1] < max_lifetimes

            # Set the bins by finding automatic bin limits for the first set of points
            if "bins" not in hist_kw:
                _, bins, _ = axis.hist(
                    arr[:, i][mask], bins=100, **hist_kw, label=label
                )
                hist_kw["bins"] = bins

            else:
                axis.hist(arr[:, i], **hist_kw, label=label)

        # Remove the bins from the dict once we've plotted all the points
        hist_kw.pop("bins")

    ax.ravel()[-1].legend()

    return fig, ax


def _parameterise(data_frame: pd.DataFrame):
    """
    Find parameterisation of a dataframe

    """
    k = np.row_stack(tuple(data_frame[k] for k in definitions.MOMENTUM_COLUMNS[0:4]))
    pi1 = np.row_stack(tuple(data_frame[k] for k in definitions.MOMENTUM_COLUMNS[4:8]))
    pi2 = np.row_stack(tuple(data_frame[k] for k in definitions.MOMENTUM_COLUMNS[8:12]))
    pi3 = np.row_stack(
        tuple(data_frame[k] for k in definitions.MOMENTUM_COLUMNS[12:16])
    )

    pi1, pi2 = util.momentum_order(k, pi1, pi2)

    return np.column_stack((helicity_param(k, pi1, pi2, pi3), data_frame["time"]))


def main():
    """
    Create a plot

    """
    # These return generators
    rs_data = _parameterise(
        util.flip_momenta(pd.concat(get.data("2018", "cf", "magdown")))
    )
    ws_data = _parameterise(
        util.flip_momenta(pd.concat(get.data("2018", "dcs", "magdown")))
    )

    rs_pgun = _parameterise(
        util.flip_momenta(get.particle_gun("cf", show_progress=True))
    )
    ws_pgun = _parameterise(
        util.flip_momenta(get.particle_gun("dcs", show_progress=True))
    )

    rs_ampgen = _parameterise(get.ampgen("cf"))
    ws_ampgen = _parameterise(get.ampgen("dcs"))

    _plot(
        [rs_data, rs_pgun, rs_ampgen, ws_data, ws_pgun, ws_ampgen],
        ["CF data", "CF pgun", "CF AmpGen", "DCS data", "DCS pgun", "DCS AmpGen"],
    )

    plt.show()


if __name__ == "__main__":
    main()
