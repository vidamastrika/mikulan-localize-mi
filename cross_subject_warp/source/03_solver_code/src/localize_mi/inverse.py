"""
Reusable inverse-modelling functions for Localize-MI.

Purpose
-------
This module contains the source-localization steps shared by later
analysis scripts:

1. Apply the average EEG reference.
2. Estimate pre-stimulation noise covariance.
3. Average the stimulation-locked epochs.
4. Select the target time interval.
5. Create the inverse operator.
6. Apply MNE, dSPM, sLORETA, or eLORETA.

Keeping these operations in one module ensures that all later
experiments use the same processing steps.

Inputs
------
The main input consists of:

1. MNE epochs loaded using ``localize_mi.io.load_run``.
2. The matching participant-specific forward model.
3. The inverse method and its parameter values.

Outputs
-------
The main ``run_inverse`` function returns an ``InverseResult`` object
containing:

1. Average-referenced epochs.
2. Noise covariance.
3. Inverse operator.
4. Averaged EEG inside the target interval.
5. Cortical source estimate.
6. All parameter values used.

This module does not save files or create figures. Individual scripts
decide which results should be saved.
"""

from dataclasses import dataclass
from typing import Any
import warnings

import mne


# These are the minimum-norm methods supported in our experiments.
# Lower-case keys make command-line input easier to handle.
SUPPORTED_METHODS = {
    "mne": "MNE",
    "dspm": "dSPM",
    "sloreta": "sLORETA",
    "eloreta": "eLORETA",
}


@dataclass
class InverseResult:
    """
    Store the complete result of one inverse calculation.

    Attributes
    ----------
    epochs : mne.Epochs
        EEG epochs after applying the average reference.
    covariance : mne.Covariance
        Pre-stimulation noise covariance.
    inverse_operator : object
        Mapping from scalp EEG to cortical sources.
    evoked : mne.Evoked
        Averaged EEG inside the selected target interval.
    source_estimate : mne.SourceEstimate
        Estimated cortical activity over time.
    method : str
        Inverse method in MNE's standard spelling.
    covariance_tmin, covariance_tmax : float
        Covariance interval in seconds.
    target_tmin, target_tmax : float
        Source-localization interval in seconds.
    loose : float
        Amount of freedom given to source orientation.
    depth : float
        Strength of depth compensation.
    snr : float
        Assumed signal-to-noise ratio.
    lambda2 : float
        Regularization value calculated from SNR.
    """

    epochs: Any
    covariance: Any
    inverse_operator: Any
    evoked: Any
    source_estimate: Any
    method: str
    covariance_tmin: float
    covariance_tmax: float
    target_tmin: float
    target_tmax: float
    loose: float
    depth: float
    snr: float
    lambda2: float


def normalize_method_name(method):
    """
    Convert a method name into the spelling expected by MNE.

    For example, ``eloreta`` becomes ``eLORETA`` and ``dspm`` becomes
    ``dSPM``.

    Parameters
    ----------
    method : str
        User-provided inverse-method name.

    Returns
    -------
    str
        Normalized method name accepted by MNE.

    Raises
    ------
    ValueError
        Raised when the requested method is unsupported.
    """

    normalized_key = (
        str(method)
        .replace("-", "")
        .replace("_", "")
        .lower()
    )

    if normalized_key not in SUPPORTED_METHODS:
        available = ", ".join(
            SUPPORTED_METHODS.values()
        )

        raise ValueError(
            f"Unsupported inverse method: {method!r}. "
            f"Available methods: {available}."
        )

    return SUPPORTED_METHODS[normalized_key]


def apply_average_reference(epochs):
    """
    Apply the average EEG reference to a copy of the epochs.

    The reference is calculated using channels that are not marked bad.
    A copy is returned so that the originally loaded epochs remain
    unchanged.

    Parameters
    ----------
    epochs : mne.Epochs
        Loaded EEG epochs.

    Returns
    -------
    mne.Epochs
        Average-referenced copy of the epochs.
    """

    referenced_epochs = epochs.copy()

    referenced_epochs.set_eeg_reference(
        ref_channels="average",
        projection=True,
        verbose=False,
    )

    referenced_epochs.apply_proj(
        verbose=False,
    )

    return referenced_epochs


def estimate_noise_covariance(
    epochs,
    tmin=-0.250,
    tmax=-0.050,
    method="auto",
    verbose=True,
):
    """
    Estimate background EEG variation before stimulation.

    Covariance describes how the EEG channels vary together during the
    pre-stimulation interval. The inverse model uses this information
    when separating background variation from stimulation-related
    activity.

    The released values were baseline-corrected before export. However,
    the complete original baseline interval is no longer available.
    Therefore, this function suppresses only MNE's metadata warning and
    does not apply baseline correction again.

    Parameters
    ----------
    epochs : mne.Epochs
        Average-referenced EEG epochs.
    tmin : float
        Beginning of the covariance interval in seconds.
    tmax : float
        End of the covariance interval in seconds.
    method : str
        Covariance estimator. ``auto`` follows the authors' example.
    verbose : bool
        Whether MNE should print calculation details.

    Returns
    -------
    mne.Covariance
        Estimated noise covariance.
    """

    if tmin < epochs.tmin:
        raise ValueError(
            f"Covariance tmin {tmin} is earlier than "
            f"the available EEG tmin {epochs.tmin}."
        )

    if tmax > epochs.tmax:
        raise ValueError(
            f"Covariance tmax {tmax} is later than "
            f"the available EEG tmax {epochs.tmax}."
        )

    if tmin >= tmax:
        raise ValueError(
            "Covariance tmin must be earlier than tmax."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Epochs are not baseline corrected.*",
            category=RuntimeWarning,
        )

        covariance = mne.compute_covariance(
            epochs,
            method=method,
            tmin=tmin,
            tmax=tmax,
            verbose=verbose,
        )

    return covariance


def create_inverse_operator(
    epochs,
    forward,
    covariance,
    loose=1.0,
    depth=0.1,
    verbose=True,
):
    """
    Create the mapping from scalp EEG to cortical sources.

    The inverse operator combines the participant-specific forward
    model, pre-stimulation covariance, and assumptions about source
    orientation and depth.

    Parameters
    ----------
    epochs : mne.Epochs
        Average-referenced EEG epochs.
    forward : mne.Forward
        Participant-specific forward model.
    covariance : mne.Covariance
        Pre-stimulation noise covariance.
    loose : float
        Orientation freedom between 0 and 1. The authors used 1.0.
    depth : float
        Depth compensation between 0 and 1. The authors used 0.1.
    verbose : bool
        Whether MNE should print calculation details.

    Returns
    -------
    object
        MNE inverse operator.
    """

    if not 0 <= loose <= 1:
        raise ValueError(
            "Loose orientation must be between 0 and 1."
        )

    if not 0 <= depth <= 1:
        raise ValueError(
            "Depth weighting must be between 0 and 1."
        )

    inverse_operator = (
        mne.minimum_norm.make_inverse_operator(
            info=epochs.info,
            forward=forward,
            noise_cov=covariance,
            loose=loose,
            depth=depth,
            verbose=verbose,
        )
    )

    return inverse_operator


def create_target_evoked(
    epochs,
    tmin=-0.002,
    tmax=0.002,
):
    """
    Average the epochs and retain the interval to localize.

    The default interval of -2 to +2 ms follows the authors' example
    and contains the immediate electrical stimulation transient.

    Parameters
    ----------
    epochs : mne.Epochs
        Average-referenced EEG epochs.
    tmin : float
        Beginning of the target interval in seconds.
    tmax : float
        End of the target interval in seconds.

    Returns
    -------
    mne.Evoked
        Averaged EEG cropped to the target interval.
    """

    if tmin < epochs.tmin:
        raise ValueError(
            f"Target tmin {tmin} is earlier than "
            f"the available EEG tmin {epochs.tmin}."
        )

    if tmax > epochs.tmax:
        raise ValueError(
            f"Target tmax {tmax} is later than "
            f"the available EEG tmax {epochs.tmax}."
        )

    if tmin >= tmax:
        raise ValueError(
            "Target tmin must be earlier than tmax."
        )

    evoked = epochs.average()

    evoked.crop(
        tmin=tmin,
        tmax=tmax,
        include_tmax=True,
    )

    return evoked


def apply_inverse_method(
    evoked,
    inverse_operator,
    method="eLORETA",
    snr=1.0,
    verbose=True,
):
    """
    Apply one inverse method to the averaged scalp EEG.

    Parameters
    ----------
    evoked : mne.Evoked
        Averaged EEG inside the target interval.
    inverse_operator : object
        Previously created inverse operator.
    method : str
        MNE, dSPM, sLORETA, or eLORETA.
    snr : float
        Assumed signal-to-noise ratio.
    verbose : bool
        Whether MNE should print calculation details.

    Returns
    -------
    source_estimate : mne.SourceEstimate
        Estimated cortical activity over time.
    normalized_method : str
        Method name in MNE's standard spelling.
    lambda2 : float
        Regularization value calculated as ``1 / SNR²``.
    """

    normalized_method = normalize_method_name(
        method
    )

    if snr <= 0:
        raise ValueError(
            "SNR must be greater than zero."
        )

    lambda2 = 1.0 / float(snr) ** 2

    source_estimate = (
        mne.minimum_norm.apply_inverse(
            evoked=evoked,
            inverse_operator=inverse_operator,
            lambda2=lambda2,
            method=normalized_method,
            pick_ori=None,
            verbose=verbose,
        )
    )

    return (
        source_estimate,
        normalized_method,
        lambda2,
    )


def run_inverse(
    epochs,
    forward,
    method="eLORETA",
    covariance_tmin=-0.250,
    covariance_tmax=-0.050,
    target_tmin=-0.002,
    target_tmax=0.002,
    loose=1.0,
    depth=0.1,
    snr=1.0,
    covariance_method="auto",
    verbose=True,
):
    """
    Run the complete shared inverse-modelling workflow.

    The function performs average referencing, covariance estimation,
    epoch averaging, inverse-operator construction, and cortical source
    estimation.

    Parameters
    ----------
    epochs : mne.Epochs
        Loaded EEG epochs.
    forward : mne.Forward
        Matching participant-specific forward model.
    method : str
        MNE, dSPM, sLORETA, or eLORETA.
    covariance_tmin, covariance_tmax : float
        Pre-stimulation covariance interval in seconds.
    target_tmin, target_tmax : float
        Source-localization interval in seconds.
    loose : float
        Orientation freedom.
    depth : float
        Depth compensation.
    snr : float
        Assumed signal-to-noise ratio.
    covariance_method : str
        Covariance estimator.
    verbose : bool
        Whether MNE should print calculation details.

    Returns
    -------
    InverseResult
        Referenced epochs, covariance, inverse operator, averaged EEG,
        source estimate, and all parameter values used.
    """

    normalized_method = normalize_method_name(
        method
    )

    referenced_epochs = apply_average_reference(
        epochs
    )

    covariance = estimate_noise_covariance(
        epochs=referenced_epochs,
        tmin=covariance_tmin,
        tmax=covariance_tmax,
        method=covariance_method,
        verbose=verbose,
    )

    inverse_operator = create_inverse_operator(
        epochs=referenced_epochs,
        forward=forward,
        covariance=covariance,
        loose=loose,
        depth=depth,
        verbose=verbose,
    )

    evoked = create_target_evoked(
        epochs=referenced_epochs,
        tmin=target_tmin,
        tmax=target_tmax,
    )

    (
        source_estimate,
        normalized_method,
        lambda2,
    ) = apply_inverse_method(
        evoked=evoked,
        inverse_operator=inverse_operator,
        method=normalized_method,
        snr=snr,
        verbose=verbose,
    )

    return InverseResult(
        epochs=referenced_epochs,
        covariance=covariance,
        inverse_operator=inverse_operator,
        evoked=evoked,
        source_estimate=source_estimate,
        method=normalized_method,
        covariance_tmin=float(covariance_tmin),
        covariance_tmax=float(covariance_tmax),
        target_tmin=float(target_tmin),
        target_tmax=float(target_tmax),
        loose=float(loose),
        depth=float(depth),
        snr=float(snr),
        lambda2=float(lambda2),
    )