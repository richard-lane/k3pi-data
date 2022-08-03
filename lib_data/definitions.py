"""
Useful definitions and things

"""
import pathlib

DUMP_DIR = pathlib.Path(__file__).resolve().parents[1] / "dumps"

AMPGEN_DIR = DUMP_DIR / "ampgen"


def ampgen_dump(sign: str) -> pathlib.Path:
    """
    Returns the location of the AmpGen dump

    sign should be "cf" or "dcs"

    """
    assert sign in {"cf", "dcs"}

    return AMPGEN_DIR / f"{sign}.pkl"
