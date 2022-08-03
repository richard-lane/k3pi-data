"""
Useful definitions and things

"""
import pathlib

DUMP_DIR = pathlib.Path(__file__).resolve().parents[1] / "dumps"

AMPGEN_DIR = DUMP_DIR / "ampgen"

# Column names for momenta
AMPGEN_MOMENTUM_SUFFICES = "Px", "Py", "Pz", "E"
MOMENTUM_COLUMNS = [
    *(f"Kplus_{s}" for s in AMPGEN_MOMENTUM_SUFFICES),
    *(f"pi1minus_{s}" for s in AMPGEN_MOMENTUM_SUFFICES),
    *(f"pi2minus_{s}" for s in AMPGEN_MOMENTUM_SUFFICES),
    *(f"pi3plus_{s}" for s in AMPGEN_MOMENTUM_SUFFICES),
]

# Particle gun directories
RS_PGUN_SOURCE_DIR = pathlib.Path(
    "/eos/lhcb/user/n/njurik/D02Kpipipi/PGun/Tuples/27165071/"
)
WS_PGUN_SOURCE_DIR = pathlib.Path(
    "/eos/lhcb/user/n/njurik/D02Kpipipi/PGun/Tuples/27165072/"
)

PGUN_DIR = DUMP_DIR / "pgun"


def ampgen_dump(sign: str) -> pathlib.Path:
    """
    Returns the location of the AmpGen dump

    sign should be "cf" or "dcs"

    """
    assert sign in {"cf", "dcs"}

    return AMPGEN_DIR / f"{sign}.pkl"


def pgun_dir(sign: str) -> pathlib.Path:
    """
    Returns the location of a directory particle gun dumps

    sign should be "cf" or "dcs"

    """
    assert sign in {"cf", "dcs"}

    return PGUN_DIR / sign


def pgun_dump(sign: str, n: int) -> pathlib.Path:
    """
    Returns the location of the n'th particle gun dump

    sign should be "cf" or "dcs"

    """
    return pgun_dir(sign) / f"{n}.pkl"
