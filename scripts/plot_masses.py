"""
For datafames where it is applicable (particle gun, full MC, real data), plot
the D0 and D* masses and their mass difference

For now - just real data

"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from lib_data import get


def _d0_mass(df: pd.DataFrame) -> np.ndarray:
    """ D0 Mass """
    return df["D0 mass"]


def _dst_mass(df: pd.DataFrame) -> np.ndarray:
    """ D* Mass """
    return df["D* mass"]


def _delta_mass(df: pd.DataFrame) -> np.ndarray:
    """ D* - D0 Mass """
    return _dst_mass(df) - _d0_mass(df)


def main():
    """
    Create plots

    """
    # These return generators
    rs_data = pd.concat(get.data("2018", "cf", "magdown"))
    ws_data = pd.concat(get.data("2018", "dcs", "magdown"))

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    bins = np.linspace(1830, 1900, 100)
    hist_kw = {"density": True, "histtype": "step"}
    ax[0].hist(_d0_mass(rs_data), bins=bins, **hist_kw, label="RS Data")
    ax[0].hist(_d0_mass(ws_data), bins=bins, **hist_kw, label="WS Data")

    bins = np.linspace(1960, 2060, 100)
    ax[1].hist(_dst_mass(rs_data), bins=bins, **hist_kw, label="RS Data")
    ax[1].hist(_dst_mass(ws_data), bins=bins, **hist_kw, label="WS Data")

    bins = np.linspace(125, 160, 100)
    ax[2].hist(_delta_mass(rs_data), bins=bins, **hist_kw, label="RS Data")
    ax[2].hist(_delta_mass(ws_data), bins=bins, **hist_kw, label="WS Data")

    ax[0].set_title("D0 Mass")
    ax[1].set_title("D* Mass")
    ax[2].set_title(r"$\Delta$ Mass")

    fig.tight_layout()

    ax[2].legend()

    plt.show()


if __name__ == "__main__":
    main()
