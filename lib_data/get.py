"""
Functions for getting the dataframes once they've been dumped

"""
import pickle
import pandas as pd

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
