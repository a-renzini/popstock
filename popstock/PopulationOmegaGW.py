import numpy as np

import bilby
import tqdm

from .util import wave_energy, omega_gw
from scipy.interpolate import interp1d
from bilby.core.utils import infer_args_from_function_except_n_args
from bilby.core.prior import Interped

class PopulationOmegaGW(object):
    def __init__(self, mass_model, redshift_model, spin_model = None, frequency_array=None):
        
        if frequency_array is not None:
            self.frequency_array=frequency_array
        else:
            self.frequency_array=np.arange(10, 2048)
        self.mass_model = mass_model
        self.redshift_model = redshift_model

        self.m1_args = [arg for arg in self.mass_model.variable_names if 'beta' not in arg]
        self.q_args = [arg for arg in infer_args_from_function_except_n_args(self.mass_model.p_q) if 'dataset' not in arg]
        self.z_args = [arg for arg in self.redshift_model.variable_names if 'dataset' not in arg]
        
        self.calculate_weights()
        
    def set_pdraws_source(self):
        p_m1q = self.calculate_p_m1q(self.proposal_samples, {key: self.fiducial_parameters[key] for key in self.m1_args + self.q_args})
        p_z = self.calculate_p_z(self.proposal_samples, {key: self.fiducial_parameters[key] for key in self.z_args})
        self.pdraws = p_m1q * p_z
        
    def calculate_p_m1q(self, samples, mass_parameters):
        if hasattr(self.mass_model, 'n_below'):
            del self.mass_model.n_below
            del self.mass_model.n_above 
        p_m1q = self.mass_model(samples, **mass_parameters)
        
        return p_m1q

    def calculate_p_z(self, samples, redshift_parameters):
        
        self.redshift_model.cached_dvc_dz = None
        p_z = self.redshift_model(samples, **redshift_parameters)
        self.redshift_model.cached_dvc_dz = None
        
        return p_z
    
    def draw_and_set_proposal_samples(self, fiducial_parameters, N_proposal_samples = 1e5):
        
        self.set_proposal_samples(fiducial_parameters, N_proposal_samples)
        self.set_pdraws_source()        
        
    def draw_mass_proposal_samples(self, m1_grid, q_grid, population_params, N):
        
        print("Drawing m1 samples")
        
        m1_prob = self.mass_model.p_m1({'mass_1': m1_grid}, **{key: population_params[key] for key in self.m1_args})
        pm1_interped = Interped(xx = m1_grid, yy = m1_prob)
        mass_1_source_samples = pm1_interped.sample(N)
        
        print("Drawing mass ratio samples")
        q_samples = []
        for m1_sample in tqdm.tqdm(mass_1_source_samples):
            if hasattr(self.mass_model, 'n_below'):
                del self.mass_model.n_below
                del self.mass_model.n_above
            pqs = self.mass_model.p_q({'mass_1': np.array([m1_sample]), 'mass_ratio': q_grid}, **{key: population_params[key] for key in self.q_args})
            pq_interped = Interped(xx = q_grid, yy = pqs)
            q_samples.append(pq_interped.sample())
        return {'mass_1': mass_1_source_samples, 'mass_ratio': np.array(q_samples)}
    
    def draw_redshift_proposal_samples(self, z_grid, population_params, N):
        
        print("Drawing redshift samples")
        z_prob = self.redshift_model({'redshift': z_grid}, **{key: population_params[key] for key in self.z_args})
        pz_interped = Interped(xx = z_grid, yy=z_prob)
        z_samples = pz_interped.sample(N)
        
        return {'redshift': z_samples}
    
    def draw_source_proposal_samples(self, fiducial_parameters, N):
        
        proposal_samples = dict()
        mass_samples = self.draw_mass_proposal_samples(self.mass_model.m1s, self.mass_model.qs, fiducial_parameters, N)
        z_grid = np.logspace(np.log10(0.0001), np.log10(self.redshift_model.z_max), 10000)
        z_samples = self.draw_redshift_proposal_samples(z_grid, fiducial_parameters, N)
        
        proposal_samples.update(mass_samples)
        proposal_samples.update(z_samples)
        
        return proposal_samples
    
    def set_proposal_samples(self, fiducial_parameters, N):
        
        self.N_proposal_samples = N
        self.fiducial_parameters = fiducial_parameters.copy()
        proposal_samples = self.draw_source_proposal_samples(self.fiducial_parameters, self.N_proposal_samples)
        proposal_samples['mass_1_detector'] = proposal_samples['mass_1'] * (1 + proposal_samples['redshift'])
        self.proposal_samples = proposal_samples.copy()

    def calculate_weights(self, Lambda=None):
        """
        Calculate weights from Lambda0 to Lambda
        """
        if Lambda is not None:
            print('calculate weights... huh... cool...')
        else:
            self.weights = np.ones((len(self.proposal_samples)))

    def calculate_wave_energies(self, waveform_duration=10, sampling_frequency=2048, waveform_approximant='IMRPhenomD', waveform_reference_frequency=25, waveform_minimum_frequency=20):
        """
        """

        # Some of the waveform generator params can be hardcoded/defaulted for ease of use
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=waveform_duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={
                "waveform_approximant": waveform_approximant,
                "reference_frequency": waveform_reference_frequency,
                "minimum_frequency": waveform_minimum_frequency,
            },
        )

        # These will need to be interpolated to match the requested frequencies
        waveform_frequencies = waveform_generator.frequency_array
        wave_energies = []

        for i in self.proposal_samples:
            inj_sample = {}
            # Generate the individual parameters dictionary for each injection
            for k in self.proposal_samples.keys():
                try:
                    inj_sample[k] = fiducial_samples[k]['content'][i]
                except:
                    inj_sample[k] = fiducial_samples[k][i]

            wave_energies.append(interp1d(waveform_frequencies, wave_energy(waveform_generator, inj_sample))(self.frequency_array))

        self.wave_energies = np.array(wave_energies)


    def calculate_omega_gw(self):
        """
        """
        self.omega_gw = omega_gw(self.frequency_array, self.wave_energies, self.weights, Rate_norm=1.e-3)
