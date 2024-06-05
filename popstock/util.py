# -*- coding: utf-8 -*-
# Copyright (C) Arianna I. Renzini 2024
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# This file is part of the popstock package.

import bilby
import numpy as np
from gwpopulation.utils import xp

from popstock.constants import H0


def wave_energy(waveform_generator, injection_parameters, use_approxed_waveform=False, inspiral_only=False):
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
    #ringdown_frequency=(1.5251 - 1.1568 ) * (c**3 / (2.0 * np.pi * G * M))
    
    try:
        #set optimal waveform duration
        waveform_generator.waveform_duration = bilby.gw.utils.calculate_time_to_merger(waveform_generator.waveform_arguments['minimum_frequency'], injection_parameters['mass_1_detector'], injection_parameters['mass_2_detector'])
    except KeyError:
        pass
    
    if use_approxed_waveform:
        orient_fact = np.cos(injection_parameters['theta_jn'])**2 + ((1+np.cos(injection_parameters['theta_jn'])**2)/2)**2
        return (orient_fact*np.abs(waveform_approx_amplitude(injection_parameters, frequencies=waveform_generator.frequency_array, inspiral_only=inspiral_only))**2)
    
    try:
        polarizations = waveform_generator.frequency_domain_strain(injection_parameters)
        # Could make this into a FrequencySeries...
        return (np.abs(polarizations['plus'])**2 + np.abs(polarizations['cross'])**2)
    except:
        print('ouch!')
        return np.zeros_like(waveform_generator.frequency_array)

    
def waveform_approx_amplitude(injection_parameters, frequencies, inspiral_only=False):
    """Ajith+, Sammut+"""
    
    # I-M-R frequencies
    from .constants import G, light_speed, m_sun, mass_to_seconds_conv
    total_mass_in_kg = float((injection_parameters['mass_1_source']+injection_parameters['mass_2_source']) * m_sun)
    total_mass_scaled = float(total_mass_in_kg * mass_to_seconds_conv)
    sym_mass_ratio = float((injection_parameters['mass_1_source']*injection_parameters['mass_2_source'])/(injection_parameters['mass_1_source']+injection_parameters['mass_2_source'])**2)
    
    f_merg = (0.29740 * sym_mass_ratio ** 2.0 + 0.044810 * sym_mass_ratio + 0.095560) / (np.pi * total_mass_scaled)
    f_ring = (0.59411 * sym_mass_ratio ** 2.0 + 0.089794 * sym_mass_ratio + 0.19111) / (np.pi * total_mass_scaled)
    f_cut = (0.84845 * sym_mass_ratio ** 2.0 + 0.12828 * sym_mass_ratio + 0.27299) / (np.pi * total_mass_scaled)
    sigma = (0.50801 * sym_mass_ratio ** 2.0 + 0.077515 * sym_mass_ratio + 0.022369) / (np.pi * total_mass_scaled)
    
    # detector frame
    f_merg /= float(1+injection_parameters['redshift'])
    f_ring /= float(1+injection_parameters['redshift'])
    f_cut /= float(1+injection_parameters['redshift'])
    sigma /= float(1+injection_parameters['redshift'])

    # NEW: cut binaries at ISCO if only inspiral phase is considered.
    # This causes the turn-off in the BNS spectrum.
    f_ISCO = 1/(6.**(3./2.)*np.pi*total_mass_scaled) / float(1+injection_parameters['redshift'])
    #

    mask_insp = frequencies < f_merg
    mask_merg = (frequencies >= f_merg) & (frequencies < f_ring)
    mask_ring = (frequencies >= f_ring) & (frequencies < f_cut)
    
    # piece-wise wave amplitude
    wave_amplitude = np.zeros(len(frequencies))
    if not inspiral_only:
        wave_amplitude[mask_insp] = (frequencies[mask_insp]/f_merg)**(-7/6)
        wave_amplitude[mask_merg] =  (frequencies[mask_merg]/f_merg)**(-2/3)
        wave_amplitude[mask_ring] = ( 1/(1 + ( (frequencies[mask_ring]-f_ring)/(sigma/2) )**2 ) )*(f_ring/f_merg)**(-2/3)
    else:
        wave_amplitude[frequencies<f_ISCO] = (frequencies[frequencies<f_ISCO]/f_merg)**(-7/6)
    
    #set zero frequency to zero just in case
    wave_amplitude[0] = 0
    from astropy.constants import kpc
    dL_in_m = float(injection_parameters['luminosity_distance']*kpc.value*1.e3)
    const = (G*total_mass_in_kg*float(1+injection_parameters['redshift']))**(5/6) * f_merg**(-7/6)/(dL_in_m)/np.pi**(2/3) * (5*sym_mass_ratio/24)**(1/2) / light_speed**(3/2)
    return const * wave_amplitude
    
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
    weighted_energy = xp.nansum(weights[:, None] * wave_energies, axis=0) / len(weights)

    return Rate_norm * conv * weighted_energy

def icdf_powerlaw(unit_value, alpha, low, high):
    oneplusalpha = 1 + alpha
    a = low**oneplusalpha
    b = high**oneplusalpha - a
    return (a + b * unit_value)**(1/oneplusalpha)


def sample_powerlaw(spectral_index, low, high, N):
    """
    This is the actual spectral index x^alpha, not the negative of the spectral index
    """
    unit_samples = np.random.random(N)
    return icdf_powerlaw(unit_samples, alpha = spectral_index, low = low, high = high)

def pdf_powerlaw(value, alpha, low, high):
    return (-alpha - 1) * (low / value)**-alpha / (low * (1 - (low/high)**(-alpha-1)))
