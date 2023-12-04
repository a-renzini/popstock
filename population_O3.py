#!/bin/env python
import sys
sys.path.append("/home/arianna.renzini/PROJECTS/popstock")

from popstock.PopulationOmegaGW import PopulationOmegaGW
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift

import argparse 
import numpy as np
import bilby
import tqdm
import json

import os
from pathlib import Path

from bilby.core.prior import Interped
from bilby.core.utils import infer_args_from_function_except_n_args

"""
***
"""

parser = argparse.ArgumentParser()
parser.add_argument('-ns', '--number_samples',help="number of samples.",action="store", type=int, default=50000)
parser.add_argument('-nt', '--number_trials',help="number of population trials.",action="store", type=int, default=10000)
parser.add_argument('-wf', '--waveform_approximant',help="Wavefrom approximant.",action="store", type=str, default='IMRPhenomD')
parser.add_argument('-rd', '--run_directory',help="Run directory.",action="store", type=str, default='./')
args = parser.parse_args()

N_proposal_samples=args.number_samples
N_trials=args.number_trials
wave_approx=args.waveform_approximant 
tag=f'{wave_approx}_{N_proposal_samples}_samples_{N_trials}_trials'
rundir=Path(args.run_directory)

"""
***
"""

Lambda_0 =  {'alpha': 2.5, 'beta': 1, 'delta_m': 3, 'lam': 0.04, 'mmax': 100, 'mmin': 4, 'mpp': 33, 'sigpp':5, 'gamma': 2.7, 'kappa': 5, 'z_peak': 1.9, 'rate': 15}

mass_obj = SinglePeakSmoothedMassDistribution()
redshift_obj = MadauDickinsonRedshift(z_max=10)

freqs = np.arange(10, 2000, 5)
newpop = PopulationOmegaGW(mass_model=mass_obj, redshift_model=redshift_obj, frequency_array=freqs)

newpop.draw_and_set_proposal_samples(Lambda_0, N_proposal_samples=N_proposal_samples)
newpop.calculate_omega_gw(waveform_approximant=wave_approx)

np.savez(f"{os.path.join(rundir, f'omegagw_0_{tag}.npz')}", omega_gw=newpop.omega_gw, freqs=newpop.frequency_array, fiducial_samples=newpop.proposal_samples, Lambda_0=Lambda_0, draw_dict=newpop.pdraws)

new_omegas={}

new_omegas['Lambdas'] = []
new_omegas['Neff'] = []
new_omegas['omega_gw'] = []

result = bilby.core.result.read_in_result(filename='/home/jacob.golomb/o3b-population-data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
lambda_samples = result.posterior.sample(N_trials).to_dict('list')

print('Running trials...')
for idx in tqdm.tqdm(range(N_trials)):
    Lambda_new = {
            'alpha': lambda_samples['alpha'][idx], 
            'beta': lambda_samples['beta'][idx], 
            'delta_m': lambda_samples['delta_m'][idx],
            'lam': lambda_samples['lam'][idx], 
            'mmax': lambda_samples['mmax'][idx],
            'mmin': lambda_samples['mmin'][idx],
            'mpp': lambda_samples['mpp'][idx],
            'sigpp': lambda_samples['sigpp'][idx],
            'rate': lambda_samples['rate'][idx],
            'gamma': lambda_samples['lamb'][idx],
            'kappa': 5,
            'z_peak': 0.3*(0.5-np.random.rand())+1.9,
            }
    new_omegas['Lambdas'].append(Lambda_new)

    newpop.calculate_omega_gw(sampling_frequency=2048, Lambda=Lambda_new)
    new_omegas['Neff'].append( (np.sum(newpop.weights)**2)/(np.sum(newpop.weights**2)) )
    new_omegas['omega_gw'].append(newpop.omega_gw.tolist())

new_omegas['freqs']=newpop.frequency_array.tolist()

omegas_dict = json.dumps(new_omegas)
f = open(f"{os.path.join(rundir, f'new_omegas_{tag}.json')}","w")
f.write(omegas_dict)
f.close()

print('Done!')

exit()
