"""
Functions for performing straight cuts

Each function returns a boolean mask of which events to keep

"""
from typing import Tuple
import numpy as np


def d0_mass(tree) -> np.ndarray:
    """
    The best fit mass of the D0 after ReFit

    """
    # Jagged array; take the first (best-fit) value for each
    return tree["Dst_ReFit_D0_M"].array()[:, 0]


def dst_mass(tree) -> np.ndarray:
    """
    Best fit mass of D* after ReFit

    """
    return tree["Dst_ReFit_M"].array()[:, 0]


def _d0_mass_keep(tree) -> np.ndarray:
    """
    Keep events near the nominal D0 mass

    """
    min_mass, max_mass = 1840.83, 1888.83
    d0_m = d0_mass(tree)

    return (min_mass < d0_m) & (d0_m < max_mass)


def _delta_m_keep(tree) -> np.ndarray:
    """
    Keep events where the D* - D0 mass difference is near the nominal pi mass

    """
    min_mass, max_mass = 139.3, 152.0

    # Jagged array - take the first (best fit) value for the D* masses
    delta_m = dst_mass(tree) - d0_mass(tree)

    return (min_mass < delta_m) & (delta_m < max_mass)


def _ipchi2(tree) -> np.ndarray:
    """
    Keep events where the IPCHI2 for the D0 is small

    """
    return tree["D0_IPCHI2_OWNPV"].array() < 9.0


def sanity_keep(tree) -> np.ndarray:
    """
    Boolean mask of events to keep after sanity cuts (D/D* mass, ipchi2)

    """
    return np.logical_and.reduce(
        [
            fcn(tree)
            for fcn in (
                _d0_mass_keep,
                _delta_m_keep,
                _ipchi2,
            )
        ]
    )


def hlt_keep_pgun(tree) -> np.ndarray:
    """
    Which events to keep using the particle gun triggers

    """
    return (tree["Dplus_Hlt1TrackMVADecision_TOS"].array() == 1) | (
        tree["Dplus_Hlt1TwoTrackMVADecision_TOS"].array() == 1
    )


def _l0_keep(tree) -> np.ndarray:
    """
    Keep any events where the global L0 trigger is TIS or hadron trigger is TOS

    """
    return (tree["D0_L0HadronDecision_TOS"].array() == 1) | (
        tree["Dst_L0Global_TIS"].array() == 1
    )


def _hlt_keep(tree) -> np.ndarray:
    """
    Keep any events where either the 1 or 2 track HLT1 decision is TOS

    """
    return (tree["D0_Hlt1TrackMVADecision_TOS"].array() == 1) | (
        tree["D0_Hlt1TwoTrackMVADecision_TOS"].array() == 1
    )


def trigger_keep(tree) -> np.ndarray:
    """
    Boolean mask of events to keep after HLT cuts

    """
    return np.logical_and(_l0_keep(tree), _hlt_keep(tree))


def bkgcat(tree) -> np.ndarray:
    """
    Keep signal events only by cutting on the D0 and D* background category

    """
    return (tree["D0_BKGCAT"].array() == 0) & (tree["Dst_BKGCAT"].array() == 0)
