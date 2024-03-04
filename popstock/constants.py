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

from astropy.constants import c
from astropy.cosmology import Planck18 as cosmo

H0 = cosmo.H(0)
light_speed = c.value
m_sun = 1.99e30 #kg
G = 6.67e-11 #N*m^2/kg^2
mass_to_seconds_conv = G/light_speed**3

# z - dL grid
import numpy as np
from scipy.interpolate import interp1d

zees = np.arange(0, 10.01, 0.01)
dells = [cosmo.luminosity_distance(z).value for z in zees]
z_to_dL_interpolant = interp1d(zees, dells)
