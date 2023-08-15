import bilby
from .util import wave_energy, omega_gw
from scipy.interpolate import interp1d


class PopulationOmegaGW(object):
    def __init__(N_samples=5000, frequencies, ):
        self.N_samples=N_samples
        self.frequencies=frequencies
        ...
        self.fiducial_samples = '... N_samples samples from fiducial population'

    def calculate_weights(self, Lambda):
        """
        Calculate weights from Lambda0 to Lambda
        """
        ...
        self.weights = ...

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

        for i in self.fiducial_samples:
        inj_sample = {}
        # Generate the individual parameters dictionary for each injection
        for k in fiducial_samples.keys():
            try:
                inj_sample[k] = fiducial_samples[k]['content'][i]
            except:
                inj_sample[k] = fiducial_samples[k][i]

	    wave_energies.append(interp1d(waveform_frequencies, wave_energy(waveform_generator, inj_sample))(self.frequencies))

	self.wave_energies = np.array(wave_energies)
	

    def calculate_omega_gw(self, )
        """
        """
        return omega_gw(self.frequencies, self.wave_energies, self.weights, T_observation=1):
