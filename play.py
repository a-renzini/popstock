from popstock.PopulationOmegaGW import PopulationOmegaGW
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift

import numpy as np
import bilby
import tqdm
import json

from bilby.core.prior import Interped
import matplotlib.pyplot as plt
from bilby.core.utils import infer_args_from_function_except_n_args

"""
***
"""

N_proposal_samples=5000
N_trials=10000
tag=f'{N_proposal_samples}_samples_{N_trials}_trials'

"""
***
"""

Lambda_0 =  {'alpha': 2.5, 'beta': 1, 'delta_m': 3, 'lam': 0.04, 'mmax': 100, 'mmin': 4, 'mpp': 33, 'sigpp':5, 'gamma': 2.7, 'kappa': 5, 'z_peak': 1.9, 'rate': 15}

mass_obj = SinglePeakSmoothedMassDistribution()
redshift_obj = MadauDickinsonRedshift(z_max=10)

freqs = np.arange(1, 2048, 0.5)
newpop = PopulationOmegaGW(mass_model=mass_obj, redshift_model=redshift_obj, frequency_array=freqs)

newpop.draw_and_set_proposal_samples(Lambda_0, N_proposal_samples=N_proposal_samples)
newpop.calculate_omega_gw()

np.savez(f'omegagw_0_{tag}.npz', omega_gw=newpop.omega_gw, freqs=newpop.frequency_array, fiducial_samples=newpop.proposal_samples, Lambda_0=Lambda_0, draw_dict=newpop.pdraws)

newpopTF2 = PopulationOmegaGW(mass_model=mass_obj, redshift_model=redshift_obj, frequency_array=freqs)

# find better way to "clone" object
newpopTF2.set_proposal_samples(fiducial_parameters=Lambda_0, proposal_samples=newpop.proposal_samples)
newpopTF2.pdraws = newpop.pdraws
newpopTF2.calculate_omega_gw(waveform_approximant='TaylorF2')

np.savez(f'omegagw_0_TF2_{tag}.npz', omega_gw=newpopTF2.omega_gw, freqs=newpopTF2.frequency_array, fiducial_samples=newpopTF2.proposal_samples, Lambda_0=Lambda_0)


new_omegas={}
new_omegasTF2={}

result = bilby.core.result.read_in_result(filename='/home/jacob.golomb/o3b-population-data/analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
lambda_samples = result.posterior.sample(N_trials).to_dict('list')

print('Running trials...')
for idx in tqdm.tqdm(range(N_trials)):
    new_omegas[f'{idx}']={}
    new_omegasTF2[f'{idx}']={}

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
    new_omegas[f'{idx}']['Lambda']=Lambda_new
    new_omegasTF2[f'{idx}']['Lambda']=Lambda_new

    newpop.calculate_omega_gw(sampling_frequency=4096, Lambda=Lambda_new)
    
    new_omegas[f'{idx}']['omega_gw']=newpop.omega_gw.tolist()
    new_omegas[f'{idx}']['freqs']=newpop.frequency_array.tolist()

    newpopTF2.calculate_omega_gw(sampling_frequency=4096, Lambda=Lambda_new)
    
    new_omegasTF2[f'{idx}']['omega_gw']=newpopTF2.omega_gw.tolist()
    new_omegasTF2[f'{idx}']['freqs']=newpopTF2.frequency_array.tolist()

omegas_dict = json.dumps(new_omegas)
f = open(f"new_omegas_{tag}.json","w")
f.write(omegas_dict)
f.close()

json_dict = json.dumps(new_omegasTF2)
f = open(f"new_omegas_TF2_{tag}.json","w")
f.write(json_dict)
f.close()

print('Done!')

exit()
