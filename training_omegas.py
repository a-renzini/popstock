#!/bin/env python
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
# This file is part of the stochmon package

import os
import sys

sys.path.append(os.getcwd())

import argparse
import json
import os
from pathlib import Path

import bilby
import numpy as np
import tqdm
from bilby.core.prior import Interped
from bilby.core.utils import infer_args_from_function_except_n_args
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift
from gwpopulation.utils import xp

from popstock.PopulationOmegaGW import PopulationOmegaGW

"""
***
Use popstock to generate a training set of Omegas from a set of 
hyper-parameters describing the redshift and mass distribution
of a population of black hole binaries.

popstock will calculate omega_GW once, and reweight the samples from
the fiducial distribution to the provided distributions.
***
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "-rd",
    "--run_directory",
    help="Run directory.",
    action="store",
    type=str,
    default="./",
)
parser.add_argument(
    "-sm", "--samples", help="Samples to use.", action="store", type=str, default=None
)
parser.add_argument(
    "-hsm", "--hyper_samples", help="Values of the hyper-parameters to use.", action="store", type=str, default=None
)
parser.add_argument(
    "-ns",
    "--number_samples",
    help="number of samples.",
    action="store",
    type=int,
    default=None,
)
parser.add_argument(
    "-t",
    "--tag",
    help="Tag to label the run.",
    action="store",
    type=str,
    default="test",
)
parser.add_argument(
    "-wf",
    "--waveform_approximant",
    help="Wavefrom approximant. Default is IMRPhenomD.",
    action="store",
    type=str,
    default="IMRPhenomD",
)
args = parser.parse_args()

"""
Unpack arguments
---

Loading in parameters from parser; frequencies are fixed here
(these can be easily customised later on by passing in a range and spacing).
"""

N_proposal_samples = args.number_samples
wave_approx = args.waveform_approximant
rundir = Path(args.run_directory)
tag = args.tag
freqs = np.arange(10, 2000, 5)

"""
Create mass and redshift distributions
---

These are *fixed* in this script. To use custom distributions, 
more development is required. Current distributions are imported
from the gwpopulation package (https://github.com/ColmTalbot/gwpopulation)

Mass distribution: Power-Law Plus Peak (PLPP)
(see App. B.2 of https://iopscience.iop.org/article/10.3847/2041-8213/abe949)

Redshift distribution: Madau-Dickinson Star Formation Rate (SFR)
(see https://arxiv.org/abs/1805.10270,
and https://arxiv.org/abs/2003.12152 for the normalisation)
"""

mass_obj = SinglePeakSmoothedMassDistribution()
redshift_obj = MadauDickinsonRedshift(z_max=10)

models = {
    "mass_model": mass_obj,
    "redshift_model": redshift_obj,
}

"""
Create the popstock object
---

To create the PopulationOmegaGW object, you just need the mass and 
redshift models, and frequencies to use in the spectra calculations. 
We can then either sample over the models, or use user-defined samples.
As sampling takes time (and is currently not optimized), it is recommended
to do it once and use a (healthy) set of samples for future calculations.
"""

newpop = PopulationOmegaGW(models=models, frequency_array=freqs)

if args.samples is not None:
    with open(args.samples) as samples_file:
        samples_dict = json.load(samples_file)
    Lambda_0 = samples_dict["Lambda_0"]
    samples_dict.pop("Lambda_0")
    if args.number_samples is None:
        args.number_samples = len(samples_dict["redshift"])
    else:
        for key in samples_dict.keys():
            samples_dict[key] = samples_dict[key][: args.number_samples]
    newpop.set_proposal_samples(proposal_samples=samples_dict)
    print(f"Using {args.number_samples} samples...")

else:
    # fiducial distribution Lambda_0, should be good to generate
    # healthy sample sets.
    Lambda_0 = {
        "alpha": 2.5,
        "beta": 1,
        "delta_m": 3,
        "lam": 0.04,
        "mmax": 100,
        "mmin": 4,
        "mpp": 33,
        "sigpp": 5,
        "gamma": 2.7,
        "kappa": 3,
        "z_peak": 1.9,
        "rate": 15,
    }
    newpop.draw_and_set_proposal_samples(
        Lambda_0, N_proposal_samples=N_proposal_samples
    )

"""
Calculate omega_GW once
---

Omega_GW is calculated once from the sample set provided. This
can take a while depending on the number of samples, but only needs
to be done once. using 10^6 samples takes about 40 minutes.
This first omega_GW is saved to file, together with the samples used
in case these weren't loaded from file.
"""

newpop.calculate_omega_gw(waveform_approximant=wave_approx, Lambda=Lambda_0)

if args.samples is not None:
    np.savez(
        f"{os.path.join(rundir, f'omegagw_0_{tag}.npz')}",
        omega_gw=newpop.omega_gw,
        freqs=newpop.frequency_array,
        Lambda_0=Lambda_0,
    )

else:
    np.savez(
        f"{os.path.join(rundir, f'omegagw_0_{tag}.npz')}",
        omega_gw=newpop.omega_gw,
        freqs=newpop.frequency_array,
        fiducial_samples=newpop.proposal_samples,
        Lambda_0=Lambda_0,
        draw_dict=newpop.pdraws,
    )

"""
Calculate omega_GW for a set of hyper-parameters (Lambda)
---

This is the main step to create the omega_GW training set.
The desired set of Lambdas should be passed in as a json file. 
In case none is passed, this script will look for a posterior file
from the LVK O3 Populations paper (arXiv 2010.14533) lying around
on CIT. All samples provided will be used.

Result omega_GWs with their respective Lambdas are saved to a json file.
"""

new_omegas = {}

new_omegas["Lambdas"] = []
new_omegas["Neff"] = []
new_omegas["omega_gw"] = []

if args.hyper_samples is not None:
    with open(args.hyper_samples) as samples_file:
        lambda_samples = json.load(samples_file)

else:
    try:
        result = bilby.core.result.read_in_result(filename='/home/jacob.golomb/o3b-population-data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
        lambda_samples = result.posterior.sample(10000).to_dict('list')
    except IOError:
        raise ValueError("No samples for Lambda hyper-parameters have been provided, nor a default found.")
N_trials = len(lambda_samples["alpha"])

print("Running trials...")
for idx in tqdm.tqdm(range(N_trials)):
    Lambda_new = {
        "alpha": lambda_samples["alpha"][idx],
        "beta": lambda_samples["beta"][idx],
        "delta_m": lambda_samples["delta_m"][idx],
        "lam": lambda_samples["lam"][idx],
        "mmax": lambda_samples["mmax"][idx],
        "mmin": lambda_samples["mmin"][idx],
        "mpp": lambda_samples["mpp"][idx],
        "sigpp": lambda_samples["sigpp"][idx],
        "rate": lambda_samples["rate"][idx],
        "gamma": lambda_samples['lamb'][idx],
        "kappa": 3.83, #make this madau-dickinson; right now these two fixed parameters are fixed but we can pass in other ones!
        "z_peak": 2.0,
    }
    new_omegas["Lambdas"].append(Lambda_new)

    newpop.calculate_omega_gw(sampling_frequency=2048, Lambda=Lambda_new)
    new_omegas["Neff"].append(
        float((xp.sum(newpop.weights) ** 2) / (xp.sum(newpop.weights ** 2)))
    )
    new_omegas["omega_gw"].append(newpop.omega_gw.tolist())

new_omegas["freqs"] = newpop.frequency_array.tolist()

omegas_dict = json.dumps(new_omegas)
f = open(f"{os.path.join(rundir, f'new_omegas_{tag}.json')}", "w")
f.write(omegas_dict)
f.close()

print("Done!")

exit()
