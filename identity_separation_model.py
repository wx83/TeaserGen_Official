import numpy as np

class IdentitySeparationModel:
    """
    Doesn't do any separation just passes the input back as output
    """
    def __init__(self):
        """
        Initialize your model here
        """
        pass
        
    @property
    def instruments(self):
        """ DO NOT CHANGE """
        return ['dialog', 'effect', 'music']

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def separate_music_file(self, mixed_sound_array, sample_rate):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """

        # input_length = len(left_mixed_arr)
        separated_music_arrays = {}
        output_sample_rates = {}
        for instrument in self.instruments:
            separated_music_arrays[instrument] = mixed_sound_array.copy()
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates
