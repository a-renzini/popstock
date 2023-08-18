import numpy as np

from .constants import H0

def wave_energy(waveform_generator, injection_parameters):
    """
    Compute the GW energy for a given waveform and set of parameters.

    Parameters
    =======
    waveform_generator: bilby waveform generator object
        Waveform generator for a specific waveform and set of time/frequency parameters.
    injection_parameters: dict
        Dictionary of individual GW parameters.

    Returns
    =======
    The wave energy spectrum in a np.array.
    """
    print(injection_parameters)
    polarizations = waveform_generator.frequency_domain_strain(injection_parameters)
    # Could make this into a FrequencySeries...
    return np.abs(polarizations['plus'])**2 + np.abs(polarizations['cross'])**2 

def omega_gw(frequencies, wave_energies, weights,  Rate_norm):
    """
    Compute Omega GW spectrum given a set of wave energy spectra and associated weights.

    Parameters
    =======
    frequencies: np.array
        Frequency array associated to the wave energy.
    wave_energies: np.array
        Array of wave energy spectra.
    weights: np.array
        Array of weights per sample.
    Rate norm: float, optional
        **TODO**

    Returns
    =======
    The wave energy spectrum in a np.array.
    """
    conv = frequencies**3 * 4. * np.pi**2 / (3 * H0.si.value**2)
    weighted_energy = np.sum(weights[:, None] * wave_energies, axis=0) / len(weights)

    return Rate_norm * conv * weighted_energy
