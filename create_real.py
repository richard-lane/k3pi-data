"""
Create dataframes for real data

There are lots of files - we will save each of these are their own DataFrame, for now

The data lives on lxplus; this script should therefore be run on lxplus.

"""
import os
import pickle
import argparse
import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib_data import definitions
from lib_data import cuts


def _add_momenta(df: pd.DataFrame, tree, keep: np.ndarray) -> None:
    """
    Read momenta into the dataframe, in place

    Reads from the tree, applies the keep mask, adds columns to df

    """
    suffices = "PX", "PY", "PZ", "PE"
    branches = (
        *(f"Dst_ReFit_D0_Kplus_{s}" for s in suffices),
        *(f"Dst_ReFit_D0_piplus_{s}" for s in suffices),
        *(f"Dst_ReFit_D0_piplus_0_{s}" for s in suffices),
        *(f"Dst_ReFit_D0_piplus_1_{s}" for s in suffices),
    )

    for branch, column in zip(branches, definitions.MOMENTUM_COLUMNS):
        # Take the first (best fit) value for each momentum
        df[column] = tree[branch].array()[:, 0][keep]


def _real_df(tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe with momenta, time and other arrays from the provided trees

    """
    df = pd.DataFrame()

    # Mask to perform straight cuts
    keep = cuts.sanity_keep(tree)

    # Combine with the mask to perform HLT cuts
    keep &= cuts.trigger_keep(tree)

    # Read momenta into the dataframe
    _add_momenta(df, tree, keep)

    # Read times into dataframe
    # 0.3 to convert from ctau to ps
    # 0.41 to convert from ps to D lifetimes
    # Take the first (best fit) value from each
    df["time"] = tree["Dst_ReFit_D0_ctau"].array()[:, 0][keep] / (0.3 * 0.41)

    # Read other variables - for e.g. the BDT cuts, kaon signs, etc.
    df["K ID"] = tree["Dst_ReFit_D0_Kplus_ID"].array()[:, 0][keep]

    # D, D* masses
    df["D0 mass"] = tree["Dst_ReFit_D0_M"].array()[:, 0][keep]
    df["D* mass"] = tree["Dst_ReFit_M"].array()[:, 0][keep]

    return df


def main(year: str, sign: str, magnetisation: str) -> None:
    """ Create a DataFrame holding real data info """
    # If the dir doesnt exist, create it
    if not definitions.DATA_DIR.is_dir():
        os.mkdir(definitions.DATA_DIR)
    if not definitions.data_dir(year, sign, magnetisation).is_dir():
        os.mkdir(definitions.data_dir(year, sign, magnetisation))

    # Iterate over input files
    tree_name = definitions.data_tree(sign)
    for path in tqdm(definitions.data_files(year, magnetisation)):
        # If the dump already exists, do nothing
        dump_path = definitions.data_dump(path, year, sign, magnetisation)
        if dump_path.is_file():
            continue

        with uproot.open(path) as data_f:
            # Create the dataframe
            dataframe = _real_df(data_f[tree_name])

        # Dump it
        with open(dump_path, "wb") as dump_f:
            pickle.dump(dataframe, dump_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump a pandas DataFrame to `dumps/`")
    parser.add_argument(
        "year",
        type=str,
        choices={"2018"},
        help="Data taking year.",
    )
    parser.add_argument(
        "sign",
        type=str,
        choices={"dcs", "cf"},
        help="Type of decay - favoured or suppressed."
        "D0->K+3pi is DCS; Dbar0->K+3pi is CF (or conjugate).",
    )
    parser.add_argument(
        "magnetisation",
        type=str,
        choices={"magdown"},
        help="magnetisation direction",
    )

    args = parser.parse_args()

    main(args.year, args.sign, args.magnetisation)
