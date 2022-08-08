"""
Create dataframes for the upper mass sideband of real data
Used when training the sig/bkg classifier

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
from lib_data import training_vars


def _bkg_keep(d_mass: np.ndarray, delta_m: np.ndarray) -> np.ndarray:
    """
    Mask of events to keep after background mass cuts

    """
    # Keep points far enough away from the nominal mass
    d_mass_width = 24
    d_mass_range = (
        definitions.D0_MASS_MEV - d_mass_width,
        definitions.D0_MASS_MEV + d_mass_width,
    )
    d_mass_mask = (d_mass < d_mass_range[0]) | (d_mass_range[1] < d_mass)

    # AND within the delta M upper mass sideband
    delta_m_mask = (152 < delta_m) & (delta_m < 157)

    return d_mass_mask & delta_m_mask


def _uppermass_df(tree) -> pd.DataFrame:
    """
    Populate a pandas dataframe information used for the classification

    """
    df = pd.DataFrame()

    # Mask to perform straight cuts
    # Usually we would do the "sanity cuts" here -
    # these are D0, delta M and ipchi2 cuts
    # In this case we just want the ipchi2
    keep = cuts._ipchi2(tree)

    # Combine with the mask to perform HLT cuts
    keep &= cuts.trigger_keep(tree)

    # Keep only events in the upper mass sideband
    d_mass = tree["Dst_ReFit_D0_M"].array()[:, 0]
    dst_mass = tree["Dst_ReFit_M"].array()[:, 0]
    keep &= _bkg_keep(d_mass, dst_mass - d_mass)

    # Store the D masses
    df["D0 mass"] = d_mass[keep]
    df["D* mass"] = dst_mass[keep]

    # Read other things that we want to keep for training the signal cut BDT
    for branch, array in zip(
        training_vars.training_var_names(), training_vars.training_var_functions()
    ):
        df[branch] = array(tree)[keep]

    # Read other variables - for e.g. the BDT cuts, kaon signs, etc.
    df["K ID"] = tree["Dst_ReFit_D0_Kplus_ID"].array()[:, 0][keep]
    df["slow pi ID"] = tree["Dst_ReFit_piplus_ID"].array()[:, 0][keep]

    return df


def main(year: str, sign: str, magnetisation: str) -> None:
    """ Create a DataFrame holding real data info """
    # If the dir doesnt exist, create it
    if not definitions.UPPERMASS_DIR.is_dir():
        os.mkdir(definitions.UPPERMASS_DIR)
    if not definitions.uppermass_dir(year, sign, magnetisation).is_dir():
        os.mkdir(definitions.uppermass_dir(year, sign, magnetisation))

    # Iterate over input files
    tree_name = definitions.data_tree(sign)
    for path in tqdm(definitions.data_files(year, magnetisation)):
        # If the dump already exists, do nothing
        dump_path = definitions.uppermass_dump(path, year, sign, magnetisation)
        if dump_path.is_file():
            continue

        with uproot.open(path) as data_f:
            # Create the dataframe
            dataframe = _uppermass_df(data_f[tree_name])

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
