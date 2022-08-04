"""
Functions for getting the dataframes once they've been dumped

"""
import glob
import pickle
import pandas as pd
from tqdm import tqdm

from . import definitions


def ampgen(sign: str) -> pd.DataFrame:
    """
    Get the AmpGen dataframe

    Sign should be cf or dcs

    """
    try:
        with open(definitions.ampgen_dump(sign), "rb") as df_f:
            return pickle.load(df_f)

    except FileNotFoundError as err:
        print("=" * 79)
        print(f"create the {sign} ampgen dump by running `create_ampgen.py`")
        print("=" * 79)

        raise err


def particle_gun(sign: str, show_progress: bool = False) -> pd.DataFrame:
    """
    Get the particle gun dataframes, concatenate them and return

    :param sign: "cf" or "dcs"
    :param show_progress: whether to display a progress bar

    """
    dfs = []
    paths = glob.glob(str(definitions.pgun_dir(sign) / "*"))

    progress_fcn = tqdm if show_progress else lambda x: x

    for path in progress_fcn(paths):
        try:
            with open(path, "rb") as df_f:
                dfs.append(pickle.load(df_f))

        except FileNotFoundError as err:
            print("=" * 79)
            print(f"create the {sign} ampgen dump by running `create_ampgen.py`")
            print("=" * 79)

            raise err

    return pd.concat(dfs)
