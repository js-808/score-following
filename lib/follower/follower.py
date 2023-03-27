### A score follower implementing a Viterbi-style decoding algorithm to get the optimal 
### transition timing of a performance through each MIDI event in the corresponding score 
from typing import * 
from scipy.stats import norm 
import numpy as np 
import librosa

class InvalidTransformException(Exception):
    """An exception raised when an unrecognized transform is passed 
    into the Score Follower"""
    pass 

class ScoreFollower:
    """A class to follow along with an audio performance in the 
    correlated score.
    
    Uses a Viterbi-style algoritm to decode the optimal sequence
    of frame-level score movements to use to follow the score.
    
    Attributes:
        chord_events (List[Tuple]): A list of chord events in the style \
            (onset_time, offset_time, [list of midi nums])
        audio_events ((num_samples, num_bins) ndarray)): A transformed \
            representation of the audio we'd like to follow
        num_samples (int) : The number of transformed audio samples 
        num_bins (int): The number of bins used in the transform 
    """
    ### CONSTANTS 
    MIDI_C1_NOTE_NUM : int = 24     # The midi representation of the note C1 
                                    #(the default minimum transform frequency)
    NUM_HARMONICS : int = 4         # The number of harmonics to replicate in 
                                    # our idealized templates
    IDEALIZED_HARMONIC_VARIANCE : int = 10    # The variance of our harmonic "spikes"
                                              # (can be tweaked as needed, should be small)
    FRAME_LEVEL_VARIANCE : int = 10   # The variance of the number of frames being spent
                                              # in the sustain phase of a note 

    ### METHODS 
    def __init__(self, chord_events: Sequence[Tuple], audio_events : np.ndarray, 
                 audio_transform : str = 'cqt', 
                 transform_min_freq : Optional[float] = None):
        """Save chord events and audio events as attributes.
        
        Also, create a plausible sequence (with self loops) of chord events 
        that we will try to time with the audio using a Viterbi-style 
        decoding/sequence fitting algorithm.

        Parameters:
        chord_events (List[Tuple]): A list of ordered chord events in the style \
            (onset_time, offset_time, [list of midi nums])
        audio_events ((num_bins, num_samples) ndarray)): A transformed \
            representation of the audio we'd like to follow
        audio_transform (str): The method used to transform the raw audio signal into \
            the format of audio_events.
        transform_min_freq (float): The minimum frequency used in the transform (if \
            None, the equivalent of the MIDI note C1 will be used)

        Raises:
            InvalidTransformException : If the audio_transform parameter is not recognized \
                (currently only implements 'cqt')
        """
        # Store the audio/chord data as attributes 
        self.audio_events : np.ndarray = audio_events.T     # Transposed to make index [timestep, bin] (more intuitive)
        self.num_samples, self.num_bins = self.audio_events.shape

        # Ensure the transform was valid, and get representative frequency bins of that transform
        # if it was 
        if audio_transform.strip().lower() == 'cqt':
            # Get the transform minimum frequency used in the CQT 
            if transform_min_freq is None:
                transform_min_freq = librosa.midi_to_hz(self.MIDI_C1_NOTE_NUM)
            # Get the frequency bins used in the CQT
            frequency_bin_means = librosa.cqt_frequencies(self.num_bins, fmin=transform_min_freq)
        else:
            msg = f"Error: Transform {audio_transform} not implemented by this score follower."
            raise InvalidTransformException(msg)
        
        # Create a (known) "hidden"-state sequence (with self loops) of the chord
        # and audio events based off of a negative-binomial distribution 
        self.chord_sequence = self._initialize_chord_events(chord_events)

        # Get idealized chord frequency bin templates, based off of a mixture of Gaussians 
        self.chord_templates = self._create_chord_templates(chord_events, frequency_bin_means)


    def _neg_bin_method_of_moments(self, mean: float, var: float) -> Tuple[int, float]:
        """Derives the NegBin(n,p) distribution using the method of moments, \
        finding the parameters p (failure probability/probability of transition \
        to next state in the markov chain) and n (the number of allowed failures/state \
        transitions) from the empirical mean and variance of the distribution.
        
        Parameters:
            mean (float): The empirical mean of the distribution
            var (float): The empirical variance of the distribution
        
        Returns:
            n (int): The number of allowed failures/non-self state transitions 
            p (float): The probability of failure/transitioning to a non-self state
        """
        n = max(int(np.ceil(mean**2 / (mean + var))), 1)    # Make sure it's a positive integer
        p = min(max(mean / (mean + var), 0), 1)             # Make sure it's in the range [0,1]
        return n, p 
    
    def _initialize_chord_events(self, chord_events: Sequence[Tuple]) -> List[Tuple]:
        """Create a sequence of chord events (tuples) of the style \
        (chord_label, chord_status, next_transition_prob, self_transition_prob), \
        where: \
            chord_label (Tuple(str)) : tuple of the form (midi_id_1, midi_id_2, ....) \
                uniquely describing the MIDI values of the sounding pitches \
            chord_status (str) : One of 'attack', 'sustain', or 'release' \
            next_transition_prob (float): The probability of the state transitioning \
                to the next sequential state \
            self_transition_prob (float): The probability of the state transitioning \
                back to itself \
        This sequence is designed to be as true to the timing of the score as possible, 
        while allowing for variance in the length of each note as needed.

        Parameters:
            chord_events (Sequence[Tuple]): A list of chord events in the style \
                (onset_time, offset_time, [list of midi nums])

        Returns:
            chord_sequence (List[Tuple]): The sequence of tuples described above, of the form \
                (chord_label, chord_status, next_transition_prob, self_transition_prob)
        """
        # Get the number of sample frames per MIDI timestep 
        min_midi_timestep = chord_events[0][0]     # First chord's onset time
        max_midi_timestep = chord_events[-1][1]    # Last chord's offset time 
        FRAMES_PER_MIDI_TIMESTEP = self.num_samples / (max_midi_timestep - min_midi_timestep)

        # For each chord, model the sustain as a negative binomial distribution.
        # Give each chord a 2-state attach phase, and an n-state sustain phase with self loops
        # (where n is the derived parameter from the negative binomial model)
        chord_sequence = []
        for chord_event in chord_events:
            # Get the number of expected frames 
            chord_MIDI_start, chord_MIDI_end, chord_label = chord_event[0], chord_event[1], tuple(chord_event[2])
            chord_MIDI_duration = chord_MIDI_end - chord_MIDI_start
            num_expected_frames = chord_MIDI_duration * FRAMES_PER_MIDI_TIMESTEP
            expected_sustain_frames = max(num_expected_frames - 2, 1)  # 2 attack phases need to be forgotten

            # Calculate the parameters for the negative binomial sustain phase model
            # Given quantities:
            #   mean = expected_sustain_frames (expected number of frames to go in the sustain phase, prescribed from score)
            #   variance = self.FRAME_LEVEL_VARIANCE (variance of number of frames - fixed a priori to indicate lack of knowledge)
            # Derived quantities:
            #   n (int): Number of frames expected to go in sustain phase (a positive integer)
            #   p (float): Probability of transition to next phase (in [0,1])
            #   q (float): Probablity of self-transition (in [0,1])
            n, p = self._neg_bin_method_of_moments(expected_sustain_frames, self.FRAME_LEVEL_VARIANCE)
            q = 1-p

            # Using these parameters, instantiate the corresponding chord sequence 
            attack_event = (chord_label, 'attack', 1, 0)    # MUST go to next phase (designed to capture the chord's start in our algorithm)
            sustain_event = (chord_label, 'sustain', p, q) 
            sequence = [attack_event] * 2 + [sustain_event] * n 
            chord_sequence.extend(sequence) 
        
        # Return the chord sequence 
        return chord_sequence

    def _create_chord_templates(self, chord_events: Sequence[Tuple], 
                                frequency_bin_means: np.ndarray) -> Dict[Tuple, np.ndarray]:
        """Creates a lookup dictionary of a mixture-of-gaussians signal template \
        for each chord, with each template assuming an idealized scenario where \
        a chord is modeled as a mixture of low-variance gaussians centered at the first \
        few harmonics of each note in the chord, with each harmonic decaying \
        in importance in the mixture model.

        Attributes:
            chord_events (Sequence[Tuple]): A list of chord events in the style \
                (onset_time, offset_time, [list of midi nums])
            frequency_bin_means ((self.num_bins,) ndarray): The mean frequency of each \
                frequency bin used in the audio transform.

        Returns:
            chord_profiles (Dict): A mapping of a MIDI chord (as a tuple) to a (num_bins,) length \
                ndarray containing the normalized intensity present in each frequency bin used \
                in the transform.
        """
        ### Inner function to get the idealized chord templates/profiles
        def total_ideal_signal(x : np.ndarray, chord : Tuple[int]) -> np.ndarray:
            """Inner function that returns the total normalized intensity of the ideal \
                signal of a given chord at each frequency given in the sequence `x`, \
                modeled as a mixture of low-variance gaussians centered at the first \
                few harmonics of each note in the chord, with each harmonic decaying \
                in importance.
            
            Parameters:
                x ((n,) ndarray): The frequencies for which to find the intensity \
                    of the total ideal signal at 
                chord (tuple(int)): A tuple containing the MIDI values of the notes \
                    sounding in the chord 

            Returns:
                total_signal ((n,) ndarray): The normalized intensity of the resulting \
                    signal at each frequency specified in `x`. 
            """
            # Get the fundamental frequencies of each MIDI note present in the chord
            fund_freqs_of_note = librosa.midi_to_hz(chord)      # Get the frequency (HZ) for each MIDI note 
                                                                   # in the chord 
            
            # Get the normalized weights (that add up to 1) of each harmonic in the mixture 
            # (each harmonic is less important than the last)
            weights = np.exp(np.array([-i/2 for i in range(self.NUM_HARMONICS)]))
            weights /= np.sum(weights)
            
            # Create the mixture 
            total_signal = np.zeros_like(x)
            for note in fund_freqs_of_note:
                harmonics = [(2**i) * note for i in range(self.NUM_HARMONICS)]
                for weight, harmonic in zip(weights, harmonics):
                    total_signal += (weight * norm.pdf(x, loc=harmonic, scale=10)) 
            
            # Normalize the resulting signal, and return it 
            return total_signal / np.sum(total_signal)
        
        ### Run the above function for each unique chord to get the idealized frequency profile 
        ### for each unique chord.
        chord_idealized_freq_vecs = {}
        for chord_event in chord_events:
            chord = tuple(chord_event[-1])      # Gets a tuple of form (note 1 MIDI ID, note 2 MIDI ID, etc.)
            if chord not in chord_idealized_freq_vecs:
                chord_idealized_freq_vecs[chord] = total_ideal_signal(frequency_bin_means, chord)
        return chord_idealized_freq_vecs
    
