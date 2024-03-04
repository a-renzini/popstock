#!/bin/env python
import sys

sys.path.append("/home/arianna.renzini/PROJECTS/popstock")

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

from popstock.PopulationOmegaGW import PopulationOmegaGW

"""
***
"""

parser = argparse.ArgumentParser()
parser.add_argument('-ns', '--number_samples',help="number of samples.",action="store", type=int, default=50000)
parser.add_argument('-nt', '--number_trials',help="number of population trials.",action="store", type=int, default=10000)
parser.add_argument('-wf', '--waveform_approximant',help="Wavefrom approximant.",action="store", type=str, default='IMRPhenomD')
parser.add_argument('-rd', '--run_directory',help="Run directory.",action="store", type=str, default='./')
parser.add_argument('-t', '--tag',help="File tag.",action="store", type=str, default='')

args = parser.parse_args()

N_proposal_samples=args.number_samples
N_trials=args.number_trials
wave_approx=args.waveform_approximant 
tag=f'{wave_approx}_{N_proposal_samples}_samples_{N_trials}_trials{args.tag}'
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

new_omegas_calculate = {}
new_omegas_reweight = {}

new_omegas_calculate['omega_gw'] = []
new_omegas_calculate['freqs'] = newpop.frequency_array.tolist()

new_omegas_reweight['Lambdas'] = []
new_omegas_reweight['Neff'] = []
new_omegas_reweight['omega_gw'] = []
new_omegas_reweight['freqs'] = newpop.frequency_array.tolist()

# result = bilby.core.result.read_in_result(filename='/home/jacob.golomb/o3b-population-data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
# lambda_samples = result.posterior.sample(N_trials).to_dict('list')
f_lam = open('/home/arianna.renzini/PROJECTS/popstock_tests/reweight_comparison/Lambda_samples_for_comparison.json',)
lambda_samples = json.load(f_lam)

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
    new_omegas_reweight['Lambdas'].append(Lambda_new)

    newpop.calculate_omega_gw(waveform_approximant=wave_approx, sampling_frequency=4096, Lambda=Lambda_new)
    Neff =  (np.sum(newpop.weights)**2)/(np.sum(newpop.weights**2)) 
    new_omegas_reweight['Neff'].append(Neff)
    new_omegas_reweight['omega_gw'].append(newpop.omega_gw.tolist())
    
    new_omega = {}
    new_omega['freqs'] = newpop.frequency_array.tolist()
    new_omega['Lambda'] = Lambda_new
    new_omega['Neff'] = Neff
    new_omega['omega_gw'] = newpop.omega_gw.tolist()
    
    omega_dict = json.dumps(new_omega)
    f = open(f"{os.path.join(rundir, f'new_omega_reweight_{idx}_{tag}.json')}","w")
    f.write(omega_dict)
    f.close()
    
omegas_dict = json.dumps(new_omegas_reweight)
f = open(f"{os.path.join(rundir, f'new_omegas_reweight_{tag}.json')}","w")
f.write(omegas_dict)
f.close()

for lamb in new_omegas_reweight['Lambdas']:
    newpop2 = PopulationOmegaGW(mass_model=mass_obj, redshift_model=redshift_obj, frequency_array=freqs)

    newpop2.draw_and_set_proposal_samples(lamb, N_proposal_samples=N_proposal_samples)
    newpop2.calculate_omega_gw(waveform_approximant=wave_approx)
    new_omegas_calculate['omega_gw'].append(newpop2.omega_gw.tolist())
    
    new_omega = {}
    new_omega['freqs'] = newpop2.frequency_array.tolist()
    new_omega['Lambda'] = lamb
    new_omega['omega_gw'] = newpop2.omega_gw.tolist()
    
    omega_dict = json.dumps(new_omega)
    f = open(f"{os.path.join(rundir, f'new_omega_calculate_{idx}_{tag}.json')}","w")
    f.write(omega_dict)
    f.close()
    
omegas_dict_2 = json.dumps(new_omegas_calculate)
f = open(f"{os.path.join(rundir, f'new_omegas_calculate_{tag}.json')}","w")
f.write(omegas_dict_2)
f.close()

print('Done!')

exit()

