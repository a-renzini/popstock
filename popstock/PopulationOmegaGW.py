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

import multiprocessing

import bilby
import gwpopulation
import numpy as np
import tqdm
from bilby.core.prior import Interped
from bilby.core.utils import infer_args_from_function_except_n_args
from scipy.interpolate import interp1d

from popstock.constants import z_to_dL_interpolant
from popstock.util import omega_gw, pdf_powerlaw, sample_powerlaw, wave_energy

gwpopulation.set_backend('numpy')
import warnings

from gwpopulation.utils import xp

REQUIRED_MODELS = ['mass', 'redshift']
SPIN_MODELS = ['a_1', 'a_2',  'cos_tilt_1', 'cos_tilt_2', 'chi_eff', 'chi_p', 'spin_tilt', 'spin_magnitude']

SPIN_GRIDS = {
    'a_1': xp.linspace(0, 1, 1000),
    'a_2': xp.linspace(0, 1, 1000),
    'cos_tilt_1': xp.linspace(-1, 1, 1000),
    'cos_tilt_2': xp.linspace(-1, 1, 1000)
}

MASS_GRIDS = {
    'mass_1': xp.linspace(1, 300, 1500), 
    'mass_ratio': xp.linspace(0.0001, 1, 1500),
    'mass_2': xp.linspace(1, 300, 1500),
    'total_mass': xp.linspace(2, 250, 1500)
}

MASS_ALPHAS = {
    'mass_ratio': 1.1,
    'total_mass': 2.0,
    'mass_2': 2.35
}

SKIPPED_KEYS = ['dataset', 'class', 'self']

class PopulationOmegaGW(object):

    def __init__(self, models, mass_coordinates = ['mass_1', 'mass_ratio'], frequency_array=None, backend='numpy'):
        """
        
        :math:`\Omega_{\\text{GW}}` population object.

        Parameters
        =======
        models: ``gwpopulation.model``
            Model object formatted as in the ``gwpopulation`` package.
        mass_coordinates: ``list``
            List of two parameters to describe binary population masses (e.g. ``[mass_1, mass_ratio]`` by default).
        frequency_array: ``array-like``
            If given, used to define the frequency array to calculate `\Omega_{\\text{GW}}(f)` for. Default is ``np.arange(10, 2048)``.
        """

        gwpopulation.set_backend(backend)
        from gwpopulation.utils import xp

        if frequency_array is not None:
            self.frequency_array=frequency_array
        else:
            self.frequency_array=np.arange(10, 2048)
        self.frequency_array_xp=xp.asarray(self.frequency_array)

        self.models = {key.split("_model")[0]: models[key] for key in models}

        for model in REQUIRED_MODELS:
            if model not in self.models:
                print(f"{model} is a required model input")
        
        self.spin_models = []
        for model in self.models:
            if model not in REQUIRED_MODELS:
                if model not in SPIN_MODELS:
                    print(f"{model} not recognized. Only {REQUIRED_MODELS} and {SPIN_MODELS} are allowed.")
                else:
                    self.spin_models.append(model)
        
        if 'chi_eff' in self.models:
            if ('spin_magnitude' in self.models) and ('spin_tilt' in self.models):
                print("Cannot use chi_eff model with both spin magnitude and tilt models. Reinitialize object with a different combination of models.")

        print("Initializing with the following models: ")
        for model in self.models:
            print(f"{model}: {self.models[model]}")

        self.model_args = dict()
        for model in self.models.keys():
            if hasattr(self.models[model], "variable_names"):
                self.model_args[model] = self.models[model].variable_names
            else:
                self.model_args[model] = [arg for arg in infer_args_from_function_except_n_args(self.models[model]) if (arg not in SKIPPED_KEYS) and (arg.lower() != model)]
        for coord in mass_coordinates:
            if coord.lower() != 'mass_1':
                self.other_mass_coord = coord
                break
        

        if self.other_mass_coord == 'mass_ratio':
            assert hasattr(self.models['mass'], "p_q")
            self.models['mass_ratio'] = self.models['mass'].p_q

        elif self.other_mass_coord == 'mass_2':
            try: 
                assert hasattr(self.models['mass'], "p_m2")
                self.models['mass_2'] = self.models['mass'].p_m2
            except AssertionError:
                warnings.warn('mass_2 selected as secondary mass parameter, but no p_m2 model passed. Setting identical model for mass_2.')
                self.models['mass_2'] = self.models['mass']
            
 
        elif self.other_mass_coord == 'total_mass':
            assert hasattr(self.models['mass'], "p_mtotal")
            self.models['total_mass'] = self.models['mass'].p_mtotal
        
        self.model_args[self.other_mass_coord] = [arg for arg in self.model_args['mass'] if arg in infer_args_from_function_except_n_args(self.models[self.other_mass_coord])]

        self.mass_coordinates = mass_coordinates
        self.wave_energies_calculated = False
    
    def calculate_probabilities(self, samples, population_parameters):
        """
        Calculate the probability of drawing the sample set from a specific population.
        
        Parameters
        =======
        samples: ``dict``
            Dictionary of binary samples :math:`\Theta`.
        population_parameters: ``dict``
            Dictionary of population hyper-parameters :math:`\Lambda`.
        """

        p_masses = self.calculate_p_masses(samples, {key: population_parameters[key] for key in self.model_args['mass']})
        p_z = self.calculate_p_z(samples, {key: population_parameters[key] for key in self.model_args['redshift']})
        p_spins = self.calculate_p_spin_models(samples, population_parameters)
        return p_masses * p_z * p_spins

    def set_pdraws_source(self):
        #why this popping?
        self.pdraws = self.proposal_samples.pop("pdraw")

    def calculate_pdraws(self, proposal_samples, fiducial_parameters):
        """
        Parameters
        =======
        proposal_samples: ``dict``
            Dictionary of binary samples :math:`\Theta`.
        fiducial_parameters: ``dict``
            Dictionary of fiducial population hyper-parameters :math:`\Lambda_0`, from which `proposal_samples` are drawn.
        """

        proposal_samples['pdraw'] = xp.array(self.calculate_probabilities(proposal_samples, fiducial_parameters))
        return proposal_samples
        
    def calculate_p_masses(self, samples, mass_parameters):
        """
        Calculate the probability of drawing a set of masses from the reference mass model.

        Parameters
        =======
        samples: ``dict``
            Dictionary of binary mass samples.
        mass_parameters: ``dict``
            Dictionary of hyper-parameters for the mass model describing the binary population.
        """
        if hasattr(self.models['mass'], '_q_interpolant'):
            del self.models['mass']._q_interpolant
        input_samples = dict()
        for key in self.mass_coordinates:
            if key + "_source" in samples.keys():
                input_samples[key] = samples[key+"_source"]
            else:
                input_samples[key] = samples[key]

        p_masses = self.models['mass'](input_samples, **mass_parameters)
        return p_masses

    def calculate_p_z(self, samples, redshift_parameters):
        """
        Calculate the probability of drawing a set of redshifts from the reference redshift model.

        Parameters
        =======
        samples: ``dict``
            Dictionary of binary redshift samples :math:`z`.
        redshift_parameters: ``dict``
            Dictionary of hyper-parameters for the redshift model describing the binary population.
        """
        
        self.models['redshift'].cached_dvc_dz = None
        p_z = self.models['redshift'](samples, **redshift_parameters)
        self.models['redshift'].cached_dvc_dz = None
        return p_z

    def calculate_p_spin_models(self, samples, parameters):
        """
        Calculate the probability of drawing a set of spins from the reference spin model.

        Parameters
        =======
        samples: ``dict``
            Dictionary of binary spin samples.
        parameters: ``dict``
            Dictionary of hyper-parameters for the spin model describing the binary population.
        """

        prob = 1

        for model in self.spin_models:
            prob *= self.models[model](samples, **{key: parameters[key] for key in self.model_args[model]})
        return prob

    def draw_and_set_proposal_samples(self, fiducial_parameters, N_proposal_samples = int(1e3), **sampling_kwargs):
        if not fiducial_parameters:
            raise ValueError("No valid parameters passed to set proposal samples.")
        self.fiducial_parameters = fiducial_parameters.copy()
        proposal_samples = self.draw_source_proposal_samples(self.fiducial_parameters, N_proposal_samples, **sampling_kwargs)
        
        self.set_proposal_samples(proposal_samples=proposal_samples)

    def draw_mass_proposal_samples(self, population_params, N, grids = MASS_GRIDS):
        samples = {key: [] for key in self.mass_coordinates}
        iteration = 0
        n_try = 100*N
                
        if 'mmin' in population_params.keys():
            low = float(population_params['mmin'])
        else:
            low = 3.0
        if 'mmax' in population_params.keys():
            high = float(population_params['mmax'])
        else:
            high = 100.0
            
        al_names = ['alpha', 'alpha1', 'alpha_1']
        
        alpha_mass = None
        for al in al_names:
            if al in population_params.keys():
                alpha_mass = 2.0 #np.abs(population_params[al])  #TODO!
        if alpha_mass is None:
            alpha_mass = 3.0
        
        while len(samples[self.mass_coordinates[0]]) < N:
            # change such that alpha changes based on efficiency?
            initial_samples = dict()            
            initial_samples['mass_1'] = xp.array(sample_powerlaw(-1*alpha_mass, 
                                        low = low,
                                        high = high,
                                        N = n_try,
                                        ) )
            
            initial_samples['p_draw']= xp.array(pdf_powerlaw(initial_samples["mass_1"],  
                                                     -1*alpha_mass, 
                                                    low = low,
                                                    high = high,
                                                ))
        

            initial_samples[self.other_mass_coord] = xp.array(sample_powerlaw(-1*MASS_ALPHAS[self.other_mass_coord],
                                                                    low = grids[self.other_mass_coord][0],
                                                                        high = grids[self.other_mass_coord][-1],
                                                                        N = n_try))
            initial_samples['p_draw'] *= xp.array(pdf_powerlaw(initial_samples[self.other_mass_coord], 
                                                                                -1*MASS_ALPHAS[self.other_mass_coord],
                                                                                low = grids[self.other_mass_coord][0],
                                                                                high = grids[self.other_mass_coord][-1]))
            
            probabilities = self.calculate_p_masses(initial_samples, {key: population_params[key] for key in self.model_args['mass']})
            M = max(probabilities / initial_samples['p_draw'])
            keep = xp.random.random(len(probabilities)) < (probabilities / (M *initial_samples['p_draw']))
            for key in samples:
                samples[key].extend(initial_samples[key][keep])

            n_kept = xp.nansum(keep)
            efficiency = n_kept/n_try
            print(f"Using {n_try}, got {len(samples[self.mass_coordinates[0]])} out of target {N} samples after iteration {iteration}")
            if efficiency<0.0008:
                # reduce sample size if it is too large... this should be made better in the future
                print(f'Efficiency: {efficiency} is too low, sampling directly.')
                if N>100000:
                    N=100000
                    print(f'Using capped number of samples: {N} to avoid lengthy computations.')
                return self.draw_mass_proposal_samples_OLD(population_params, N, grids = MASS_GRIDS)
            else: 
                n_try = int((N - n_kept)/efficiency)
                print(f"Efficiency: {efficiency}, trying {n_try} samples in next iteration")
                print("-----------------------------------------------------------")
                iteration += 1

        mass_proposal_samples =  {'mass_1_source': xp.array(samples['mass_1'])[:N], self.other_mass_coord: xp.array(samples[self.other_mass_coord])[:N]}
        return mass_proposal_samples

    def draw_mass_proposal_samples_indipendent(self, population_params, N, grids = MASS_GRIDS):
        
        print("Drawing m1 and m2 samples")
        prob = self.models['mass']({'mass_1': grids['mass_1'], 'mass_2': grids['mass_2']}, **{key: population_params[key] for key in self.model_args['mass']})
        prob_interped = Interped(xx = grids['mass_1'], yy = prob)
        mass_1_samples = prob_interped.sample(N)
        prob_interped = Interped(xx = grids['mass_2'], yy = prob)
        mass_2_samples = prob_interped.sample(N)
        return {'mass_1_source': mass_1_samples, 'mass_2_source': xp.array(mass_2_samples)}
    
    def draw_mass_proposal_samples_OLD(self, population_params, N, grids = MASS_GRIDS):
        
        print("Drawing m1 samples")
        
        print(type(self.models['mass']).__name__)
        m1_keys, q_keys = get_mass_keys_for_sampling(population_params, type(self.models['mass']).__name__)
        m1_prob = self.models['mass'].p_m1({'mass_1': grids['mass_1']}, **m1_keys)
        pm1_interped = Interped(xx = grids['mass_1'], yy = m1_prob)
        mass_1_source_samples = pm1_interped.sample(N)
        
        print(f"Drawing {self.other_mass_coord} samples")
        other_mass_samples = []
        for m1_sample in tqdm.tqdm(mass_1_source_samples):
            #if hasattr(self.models['mass'], 'n_below'):
            #    del self.models['mass'].n_below
            #    del self.models['mass'].n_above
            #probs = self.models[self.other_mass_coord]({'mass_1': xp.array([m1_sample]), self.other_mass_coord: grids[self.other_mass_coord]}, **{key: population_params[key] for key in self.model_args[self.other_mass_coord]})
            if hasattr(self.models['mass'], '_q_interpolant'):
                del self.models['mass']._q_interpolant
            probs = self.models[self.other_mass_coord]({'mass_1': np.array([m1_sample]), self.other_mass_coord: grids[self.other_mass_coord]}, **q_keys)
            mass_interped = Interped(xx = grids[self.other_mass_coord], yy = probs)
            other_mass_samples.append(mass_interped.sample())
        return {'mass_1_source': mass_1_source_samples, self.other_mass_coord: xp.array(other_mass_samples)}
    
    def draw_redshift_proposal_samples(self, z_grid, population_params, N):
        
        print("Drawing redshift samples")
        z_prob = self.models['redshift']({'redshift': z_grid}, **{key: population_params[key] for key in self.model_args['redshift']})
        pz_interped = Interped(xx = z_grid, yy=z_prob)
        z_samples = pz_interped.sample(N)
        
        return {'redshift': z_samples, 'luminosity_distance': z_to_dL_interpolant(z_samples)}
    
    def draw_spin_parameter_proposal_samples(self, population_params, N, grids = SPIN_GRIDS):

        spin_samples = dict()

        if len(self.spin_models) > 0:   
            ind_spins = False
            for mod in self.spin_models:
                if 'a_1' in mod:
                    ind_spins = True
            
            ind_tilts = False
            for mod in self.spin_models:
                if 'tilt_1' in mod:
                    ind_tilts = True

            if ind_spins:
                for magnitude in ['a_1', 'a_2']:
                    probs = self.models[magnitude](grids, **{key: population_params[key] for key in self.model_args[magnitude]})
                    magnitudes_interped = Interped(xx = grids[magnitude], yy = probs)
                    spin_samples[magnitude] = magnitudes_interped.sample(N)
            else:
                spin_samples.update(self.draw_spin_magnitude_proposal_samples(population_params, N))
            
            if ind_tilts:
                for tilt in ['cos_tilt_1', 'cos_tilt_2']:
                    if tilt in self.spin_models:
                        probs = self.models[tilt](grids, **{key: population_params[key] for key in self.model_args[tilt]})
                        tilts_interped = Interped(xx = grids[tilt], yy = probs)
                        spin_samples[tilt] = tilts_interped.sample(N)
            else:
                try: spin_samples.update(self.draw_spin_tilt_proposal_samples(population_params, N))
                except KeyError:
                    pass
        return spin_samples
    
    def draw_spin_magnitude_proposal_samples(self, population_params, N, grids=SPIN_GRIDS):
        prob = self.models['spin_magnitude']({'a_1': grids['a_1'], 'a_2': grids['a_2']}, **{key: population_params[key] for key in self.model_args['spin_magnitude']})
        prob_interped = Interped(xx = grids['a_1'], yy = prob)
        a_1_samples = prob_interped.sample(N)
        prob_interped = Interped(xx = grids['a_2'], yy = prob)
        a_2_samples = prob_interped.sample(N)
        return {'a_1': a_1_samples, 'a_2': xp.array(a_2_samples)}
    
    def draw_spin_tilt_proposal_samples(self, population_params, N, grids=SPIN_GRIDS):
        prob = self.models['spin_tilt']({'cos_tilt_1': grids['cos_tilt_1'], 'cos_tilt_2': grids['cos_tilt_2']}, **{key: population_params[key] for key in self.model_args['spin_tilt']})
        prob_interped = Interped(xx = grids['cos_tilt_1'], yy = prob)
        t_1_samples = prob_interped.sample(N)
        prob_interped = Interped(xx = grids['cos_tilt_2'], yy = prob)
        t_2_samples = prob_interped.sample(N)
        return {'cos_tilt_1': t_1_samples, 'cos_tilt_2': xp.array(t_2_samples)}

    def draw_source_proposal_samples(self, fiducial_parameters, N, **sampling_kwargs):
        
        proposal_samples = dict()
        mgrids = MASS_GRIDS.copy()
        try:
            mgrids['mass_1'] = self.models['mass'].m1s
            if self.other_mass_coord == 'mass_2': #this would make sense, right?
                mgrids['mass_2'] = self.models['mass'].m1s
        except AttributeError:
            pass

        try: 
            if sampling_kwargs['mass']=='direct':
                mass_samples = self.draw_mass_proposal_samples_OLD(fiducial_parameters, N, grids=mgrids)
            elif sampling_kwargs['mass']=='indipendent':
                mgrids['mass_2'] = sampling_kwargs['mass_2_grid']
                mass_samples = self.draw_mass_proposal_samples_indipendent(fiducial_parameters, N, grids=mgrids)
        except KeyError:
            mass_samples = self.draw_mass_proposal_samples(fiducial_parameters, N, grids=mgrids)
            if len(mass_samples['mass_1_source']) != N:
                N = len(mass_samples['mass_1_source']) #make sure length change goes through
        z_grid = xp.logspace(xp.log10(0.0001), xp.log10(self.models['redshift'].z_max), 10000)
        z_samples = self.draw_redshift_proposal_samples(z_grid, fiducial_parameters, N)

        spin_samples = self.draw_spin_parameter_proposal_samples(fiducial_parameters, N)
        
        proposal_samples.update(mass_samples)
        proposal_samples.update(z_samples)
        proposal_samples.update(spin_samples)
        proposal_samples.update(self.calculate_pdraws(proposal_samples, fiducial_parameters))
        
        return proposal_samples
    
    def set_proposal_samples(self, proposal_samples=None):
        
        keys = proposal_samples.keys()
        for key in proposal_samples:
            if type(proposal_samples[key]) is list:
                proposal_samples[key] = xp.array(proposal_samples[key])
        
        # fix mass samples
        if 'mass_1' in keys:
            proposal_samples['mass_1_detector'] = proposal_samples['mass_1']
            proposal_samples['mass_1_source'] = proposal_samples['mass_1_detector'] / (1 + proposal_samples['redshift'])
            if 'mass_2' in keys:
                proposal_samples['mass_2_detector'] = proposal_samples['mass_2']
            elif 'mass_ratio' in keys:
                proposal_samples['mass_2_detector'] = proposal_samples['mass_1']*proposal_samples['mass_ratio']
            else:
                raise ValueError('Missing mass parameter in proposal sample set.')
            proposal_samples['mass_2_source'] = proposal_samples['mass_2_detector'] / (1 + proposal_samples['redshift'])
        elif 'mass_1_source' in keys:
            proposal_samples['mass_1_detector'] = proposal_samples['mass_1_source'] * (1 + proposal_samples['redshift'])
            if 'mass_ratio' in keys:
                proposal_samples['mass_2_source'] = proposal_samples['mass_1_source'] * proposal_samples['mass_ratio']
            try: 
                proposal_samples['mass_2_detector'] = proposal_samples['mass_2_source'] * (1 + proposal_samples['redshift'])
            except ValueError:
                raise ValueError('Missing mass parameter in proposal sample set.')
        else:
            raise ValueError('Missing mass parameter in proposal sample set.')
        self.proposal_samples = proposal_samples.copy()
        self.N_proposal_samples = len(proposal_samples['pdraw'])

        self.set_pdraws_source()
        
        self.proposal_samples_set = True

    def calculate_weights(self, Lambda=None):
        """
        Calculate weights from Lambda0 to Lambda
        """
        if Lambda is not None:
            probabilities = self.calculate_probabilities(self.proposal_samples, Lambda)
            self.weights = (probabilities / self.pdraws)
            self.weights[np.where(probabilities==0.0)]=0.0
            self.weights[np.where(probabilities<0.0)]=0.0
        else:
            self.weights = np.ones(self.N_proposal_samples)

    def calculate_wave_energies(self, waveform_duration=10, sampling_frequency=4096, waveform_approximant='IMRPhenomD', waveform_reference_frequency=25, waveform_minimum_frequency=10, waveform_pn_phase_order=-1, multiprocess=True):
        """
        """

        # setting up internal variables for wf generation...
        self.waveform_generator = bilby.gw.WaveformGenerator(
            duration=waveform_duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={
                "waveform_approximant": waveform_approximant,
                "reference_frequency": waveform_reference_frequency,
                "minimum_frequency": waveform_minimum_frequency,
                "pn_phase_order": waveform_pn_phase_order,
            },
        )

        samples_list = self._reformatted_sample_dict(multiprocess)

        wave_energies = []

        if multiprocess:
            print('Using multiprocessing, no status bar currently supported... ')
            pool = multiprocessing.Pool()
            wave_energies = pool.starmap(get_wave_en, samples_list)
        else:
            wave_energies = []
            for inj_sample in tqdm.tqdm(samples_list):
                wave_energies.append(get_wave_en(inj_sample, self.waveform_generator, self.frequency_array))

        self.wave_energies = xp.array(wave_energies)
        self.wave_energies_calculated = True

    def to_cupy(self):
        import cupy as cp
        gwpopulation.set_backend('cupy')
        from gwpopulation.utils import xp
        if self.wave_energies_calculated:
            self.wave_energies = cp.asarray(self.wave_energies)
        if self.proposal_samples_set:
            for key in self.proposal_samples.keys():
                self.proposal_samples[key] = cp.asarray(self.proposal_samples[key])

    def calculate_omega_gw(self, Lambda=None, Rate_norm=None, multiprocess=True, **kwargs):
        """
        """
        
        if not self.wave_energies_calculated:
            self.calculate_wave_energies(multiprocess=multiprocess, **kwargs)

        self.calculate_weights(Lambda=Lambda)    
        
        if Rate_norm is None:
            
            if Lambda is None:
                Lambda = self.fiducial_parameters
            
            redshift_model_norm_in_Gpc3 = self.models['redshift'].normalisation(Lambda)/1.e9
            Rate_norm_in_Gpc3_per_seconds = Lambda['rate']/(60*60*24*365)
            Rate_norm = redshift_model_norm_in_Gpc3 * Rate_norm_in_Gpc3_per_seconds

        frequencies = self.frequency_array_xp
        self.omega_gw = omega_gw(frequencies, self.wave_energies, self.weights, Rate_norm=Rate_norm)

    def _get_wave_en(self, inj_sample):
        if not 'phase' in inj_sample:
            inj_sample['phase']=2*np.pi*np.random.rand()
        if not 'theta_jn' in inj_sample:
            cos_inc = np.random.rand()*2.-1.
            inj_sample['theta_jn']=np.arccos(cos_inc)
        if not 'a_1' in inj_sample:
            inj_sample['a_1']=0
        if not 'a_2' in inj_sample:
            inj_sample['a_2']=0
        if not 'tilt_1' in inj_sample:
            inj_sample['tilt_1']=0
        if not 'tilt_2' in inj_sample:
            inj_sample['tilt_2']=0
        use_approxed_waveform=False
        if self.waveform_approximant=='PC_waveform':
            use_approxed_waveform=True
        '''
        waveform_frequencies = xp.asarray(waveform_frequencies)
        wave_en = xp.asarray(wave_energy(waveform_generator, inj_sample, use_approxed_waveform=use_approxed_waveform))
        wave_energies.append(xp.interp(self.frequency_array, waveform_frequencies, wave_en) )
        '''
        waveform_frequencies = waveform_generator.frequency_array # These will need to be interpolated to match the requested frequencies
        wave_en = wave_energy(self.waveform_generator, inj_sample, use_approxed_waveform=use_approxed_waveform, inspiral_only=inspiral_only)
        #could also do cubic interp but takes a bit longer
        #wave_energies.append(interp1d(waveform_frequencies, wave_en, fill_value=0, bounds_error=False, kind='cubic')(frequency_array))
        return np.interp(self.frequency_array, waveform_frequencies, wave_en)

    def _reformatted_sample_dict(self, multiprocess=True):
        inj_samples = []
        for i in range(self.N_proposal_samples):
            inj_sample = {}
            # Generate the individual parameters dictionary for each injection
            for key in self.proposal_samples.keys():                
                inj_sample[key] = self.proposal_samples[key][i]
            if multiprocess:
                inj_samples.append([inj_sample, self.waveform_generator, self.frequency_array])    
            else:
                inj_samples.append(inj_sample)    
        return inj_samples

def get_wave_en(inj_sample, waveform_generator, frequency_array):
    if not 'phase' in inj_sample:
        inj_sample['phase']=2*np.pi*np.random.rand()
    if not 'theta_jn' in inj_sample:
        cos_inc = np.random.rand()*2.-1.
        inj_sample['theta_jn']=np.arccos(cos_inc)
    if not 'a_1' in inj_sample:
        inj_sample['a_1']=0
    if not 'a_2' in inj_sample:
        inj_sample['a_2']=0
    if not 'tilt_1' in inj_sample:
        inj_sample['tilt_1']=0
    if not 'tilt_2' in inj_sample:
        inj_sample['tilt_2']=0
    
    use_approxed_waveform=False
    inspiral_only=False
    if waveform_generator.waveform_arguments['waveform_approximant']=='PC_waveform':
        use_approxed_waveform=True
    elif waveform_generator.waveform_arguments['waveform_approximant']=='PC_waveform_inspiral_only':
        use_approxed_waveform=True
        inspiral_only=True
    '''
    waveform_frequencies = xp.asarray(waveform_frequencies)
    wave_en = xp.asarray(wave_energy(waveform_generator, inj_sample, use_approxed_waveform=use_approxed_waveform))
    wave_energies.append(xp.interp(self.frequency_array, waveform_frequencies, wave_en) )
    '''
    waveform_frequencies = waveform_generator.frequency_array # These will need to be interpolated to match the requested frequencies
    wave_en = wave_energy(waveform_generator, inj_sample, use_approxed_waveform=use_approxed_waveform, inspiral_only=inspiral_only)
    #could also do cubic interp but takes a bit longer
    #wave_energies.append(interp1d(waveform_frequencies, wave_en, fill_value=0, bounds_error=False, kind='cubic')(frequency_array))
    return np.interp(frequency_array, waveform_frequencies, wave_en)

def get_mass_keys_for_sampling(population_params, mass_model_name):
    m1_params = {}
    q_params = {}
    if mass_model_name == 'TwoPeakBrokenPowerLawSmoothedMassDistribution':
        m1_keys = ['alpha_1' ,'alpha_2', 'mmin', 'mmax' , 'break_mass', 'lam_0' , 'lam_1' , 'mpp_1' , 'mpp_2','delta_m', 'sigpp_1', 'sigpp_2']
        for key in m1_keys:
            try: 
                m1_params[key] = population_params[key]
            except KeyError:
                if key=='mmin':
                    m1_params['mmin'] = population_params['mlow_1']
                    q_params['mmin'] = population_params['mlow_2']
                elif key=='delta_m':
                    m1_params['delta_m'] = population_params['delta_m_1']
                    q_params['delta_m'] = population_params['delta_m_2']
                else: raise KeyError()
    q_params['beta']=population_params['beta']
    return m1_params, q_params

