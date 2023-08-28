from astropy.cosmology import Planck15 as cosmo

H0 = cosmo.H(0)
light_speed = 2.998e8
m_sun = 1.99e30 #kg
G = 6.67e-11 #N*m^2/kg^2
mass_to_seconds_conv = G/light_speed**3
