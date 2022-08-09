"""
"""
import pathlib
from typing import List

D0_MASS_MEV = 1864.84

DUMP_DIR = pathlib.Path(__file__).resolve().parents[1] / "dumps"


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

AMPGEN_DIR = DUMP_DIR / "ampgen"
PGUN_DIR = DUMP_DIR / "pgun"
MC_DIR = DUMP_DIR / "mc"
DATA_DIR = DUMP_DIR / "data"
UPPERMASS_DIR = DUMP_DIR / "upper_mass"


def data_tree(sign: str) -> str:
    """
    Name of the tree in real data

    """
    assert sign in {"cf", "dcs"}
    return (
        "Hlt2Dstp2D0Pip_D02KmPimPipPip_Tuple/DecayTree"
        if sign == "cf"
        else "Hlt2Dstp2D0Pip_D02KpPimPimPip_Tuple/DecayTree"
    )


def ampgen_dump(sign: str) -> pathlib.Path:
    """
    Returns the location of the AmpGen dump

    sign should be "cf" or "dcs"

    """
    assert sign in {"cf", "dcs"}

    return AMPGEN_DIR / f"{sign}.pkl"


def mc_files(year: str, magnetisation: str, sign: str) -> List[str]:
    """
    Paths to MC analysis productions on lxplus (or the grid maybe).

    :param year: data taking year
    :param magnetisation: "magup" or "magdown"
    :param sign: "cf" or "dcs"
    :returns: list of paths as strings

    """
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}
    assert sign in {"cf", "dcs"}

    # File holding locations of productions
    pfn_file = (
        pathlib.Path(__file__).resolve().parents[1]
        / "production_locations"
        / "mc"
        / f"{sign}_{year}_{magnetisation}.txt"
    )

    with open(pfn_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def mc_dump(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Returns the location of the MC dump

    sign should be "cf" or "dcs"

    """
    assert sign in {"cf", "dcs"}

    return MC_DIR / f"{year}_{sign}_{magnetisation}.pkl"


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


def data_dir(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Returns the location of a directory of real data dumps

    :param sign: "cf" or "dcs"
    :param year: data taking year
    :param magnetisation: "magup" or "magdown"
    :returns: absolute path to the dir used for storing this data

    """
    assert sign in {"cf", "dcs"}
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}

    return DATA_DIR / f"{year}_{sign}_{magnetisation}"


def data_files(year: str, magnetisation: str) -> List[str]:
    """
    Paths to real data analysis productions on lxplus (or the grid maybe).

    :param year: data taking year
    :param magnetisation: "magup" or "magdown"
    :returns: list of paths as strings

    """
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}

    # File holding locations of productions
    pfn_file = (
        pathlib.Path(__file__).resolve().parents[1]
        / "production_locations"
        / "real_data"
        / f"{year}_{magnetisation}_dd.txt"
    )

    with open(pfn_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def data_dump(data_file: str, year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Paths to the pickle dump corresponding to a data file.
    Idea is to pass a file returned from `data_files()` as `data_file`

    :param data_file: location of analysis production, e.g. as returned by data_files()
    :param year: data taking year
    :param sign: "cf" or "dcs"
    :param magnetisation: "magup" or "magdown"
    :returns: path to the dump location

    """
    assert sign in {"cf", "dcs"}
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}

    data_file = pathlib.Path(data_file)

    return data_dir(year, sign, magnetisation) / f"{data_file.with_suffix('').name}.pkl"


def uppermass_dir(year: str, sign: str, magnetisation: str) -> pathlib.Path:
    """
    Returns the location of a directory of real data upper mass sideband dumps

    :param sign: "cf" or "dcs"
    :param year: data taking year
    :param magnetisation: "magup" or "magdown"
    :returns: absolute path to the dir used for storing this data

    """
    assert sign in {"cf", "dcs"}
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}

    return UPPERMASS_DIR / f"{year}_{sign}_{magnetisation}"


def uppermass_dump(
    data_file: str, year: str, sign: str, magnetisation: str
) -> pathlib.Path:
    """
    Paths to the pickle dump corresponding to a the upper mass sideband of a data file.
    Idea is to pass a file returned from `data_files()` as `data_file`

    :param data_file: location of analysis production, e.g. as returned by data_files()
    :param year: data taking year
    :param sign: "cf" or "dcs"
    :param magnetisation: "magup" or "magdown"
    :returns: path to the dump location

    """
    assert sign in {"cf", "dcs"}
    assert year in {"2018"}
    assert magnetisation in {"magup", "magdown"}

    data_file = pathlib.Path(data_file)

    return (
        uppermass_dir(year, sign, magnetisation)
        / f"{data_file.with_suffix('').name}.pkl"
    )
