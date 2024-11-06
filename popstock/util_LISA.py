import numpy as np
from popstock.util import sample_powerlaw
from astropy import constants
from popstock.constants import G, light_speed, m_sun, mass_to_seconds_conv

def freq_mask(frequencies):
    freq_enter = np.array(sample_powerlaw(-11/3, low = 1.e-4, high = 0.1, N = 1))
    df = frequencies[1] - frequencies[0]
    return (frequencies > freq_enter-df/2.) & (frequencies < freq_enter+df/2.)

def dfdtCBC(mu, Mtot, f, z):
    Mtot *= (1+z)
    pref = 96 * G**(5/3) * (2*np.pi)**(8/3) /5 / light_speed**5
    return np.array(pref * mu * Mtot**(5/3) * (f)**(11/3))

def f_exit(mu, Mtot, f_enter, T_obs_in_years):
    pref = 96 * mu * (Mtot*G)**(5/3) * (2*np.pi)**(8/3) /5 / light_speed**5
    T_obs_in_seconds = T_obs_in_years * 365 * 24 * 60 * 60
    f_exit = np.power(f_enter**(-8./3.) - 8./3. * T_obs_in_seconds * pref, -3/8)
    if np.isnan(f_exit):
        return np.inf
    else:
        return f_exit
