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
        chord_events (List[Tuple]): A list of chord events from the score in \
            in the style (onset_time, offset_time, [list of midi nums])
        audio_events ((num_samples, num_bins) ndarray)): A transformed \
            representation of the audio we'd like to follow
        num_samples (int) : The number of transformed audio samples 
        num_bins (int): The number of bins used in the transform 
        chord_event_signals((len(chord_events), num_bins) ndarray):  A matrix \
            where the i-th row gives the normalized frequency spectrum expected \
            from the i-th chord in chord_events, with amplitudes reported across \
            all of the num_bins bins. 
    """
    ### ---------------- CONSTANTS ---------------- ###
    MIDI_C1_NOTE_NUM : int = 24     # The midi representation of the note C1 
                                    #(the default minimum transform frequency)
    NUM_HARMONICS : int = 4         # The number of harmonics to replicate in 
                                    # our idealized templates
    IDEALIZED_HARMONIC_VARIANCE : int = 10    # The variance of our harmonic "spikes"
                                              # (can be tweaked as needed, should be small)
    FRAME_LEVEL_VARIANCE : int = 10   # The variance of the number of frames being spent
                                              # in the sustain phase of a note 

    ### ----------- CONSTRUCTOR METHODS ----------- ###
    def __init__(self, chord_events: Sequence[Tuple], 
                 audio_events : np.ndarray, 
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
        chord_sequence = self._initialize_chord_events(chord_events)

        # Initialize a transition matrix for this state matrix 
        self.A, self.chord_transition_lookup = self._create_transition_matrix(chord_sequence)

        # Get idealized chord frequency bin templates, based off of a mixture of Gaussians 
        chord_templates = self._create_chord_templates(chord_events, frequency_bin_means)
        
        # Populate a matrix where the i-th row contains the normalized frequency
        # spectrum expected when we play the i-th chord in self.chord-sequence 
        self.chord_template_signals = np.zeros((len(chord_sequence), self.num_bins))
        for i,chord_event in enumerate(chord_sequence):
            chord = chord_event[0]
            self.chord_template_signals[i,:] = chord_templates[chord]

    def _neg_bin_method_of_moments(self, 
                                   mean: float, 
                                   var: float) -> Tuple[int, float]:
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
    
    def _initialize_chord_events(self, 
                                 chord_events: Sequence[Tuple]) -> List[Tuple]:
        """Create a sequence of chord events (tuples) of the style \
        (chord_label, chord_status, next_transition_prob, self_transition_prob), \
        where: \
            chord_label (Tuple(str)) : tuple of the form (midi_id_1, midi_id_2, ....) \
                uniquely describing the MIDI values of the sounding pitches \
            chord_status (str) : One of 'attack', 'sustain', or 'rest' \
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
        
        # Append silence of a very high variance at the beginning and end of the piece 
        silence_mean_number_of_frames = 20 
        silence_variance = 50 
        n, p = self._neg_bin_method_of_moments(silence_mean_number_of_frames, silence_variance)
        silence_event = ((), 'rest', p, 1-p)
        silence_sequence = [silence_event] * n
        chord_sequence = silence_sequence + chord_sequence + silence_sequence 

        # Return the chord sequence 
        return chord_sequence
    
    def _create_transition_matrix(self, chord_sequence : Sequence[Tuple]) -> Tuple[np.ndarray, Dict[str, Tuple]]:
        """Creates the transition matrix for the given chord sequence.
        
        Parameters:
            chord_sequence ((n,) List[Tuple]): The sequence of tuples described above, of the form \
                (chord_label, chord_status, next_transition_prob, self_transition_prob)
        
        Returns:
            A ((n,n) ndarray): The matrix of transition probabilities from
                state to state across the state space.
            index_lookup (Dict[int, Tuple]): A mapping from the index in the 
                transition matrix to the chord information from chord_sequence
        """
        # Initialize probability array
        n = len(chord_sequence)
        A = np.zeros((n,n))

        # Populate probability array in order 
        index_lookup = {}     # Mapping from index to chord it represents
        for i in range(n-1):
            chord = chord_sequence[i]
            next_transition_prob, self_transition_prob = chord[-2], chord[-1]
            A[i,i] = self_transition_prob 
            A[i+1,i] = next_transition_prob     # recall, i+1,i means probability of i going to i+1
            index_lookup[i] = chord 
        A[-1,-1] = 1        # Infinite self loop at end (for silence)
        
        # Return the desired information
        return A, index_lookup 

    def _create_chord_templates(self, 
                                chord_events: Sequence[Tuple], 
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
            
            # Normalize the resulting signal if it isn't all zeros, and return it 
            if not np.allclose(total_signal, np.zeros_like(x)):
                total_signal /=  np.sum(total_signal)
            return total_signal
        
        ### Run the above function for each unique chord to get the idealized frequency profile 
        ### for each unique chord.
        chord_idealized_freq_vecs = {}
        for chord_event in chord_events:
            chord = tuple(chord_event[-1])      # Gets a tuple of form (note 1 MIDI ID, note 2 MIDI ID, etc.)
            if chord not in chord_idealized_freq_vecs:
                chord_idealized_freq_vecs[chord] = total_ideal_signal(frequency_bin_means, chord)
        
        ### Ensure silence is part of these templates 
        chord_idealized_freq_vecs[()] = total_ideal_signal(frequency_bin_means, ())
        return chord_idealized_freq_vecs

    
    ### ----------- CALLABLE METHODS ----------- ###
    def _find_spectrum_log_prob(self, signal: np.ndarray) -> float:
        """Find the l2-norm error of the signal to the "ideal" spectrum
        of the given chord. Return something inversely proportional to the
        l2-norm error (the lower the error, the higher the probability).
        
        Parameters:
            signal ((self.num_bins, ) ndarray) : An array representing the amplitude of the \
                input signal in different frequency bins, normalized so they sum to 1
            
        Returns:
            err ((num_chords_in_sequence, ) ndarray): log(e^{-(err)}) = -err, where err is 
                the l2-error of the signal with each row (idealized signal) of \
                self.chord_event_signals
        """
        normalized_signal = signal / np.sum(signal)

        # Take the l2-norm of the difference between each of the ideal signal rows 
        # against the actual signal.
        l2_err = np.linalg.norm(self.chord_template_signals - normalized_signal, axis=1)

        # We want small errors to be good, and big errors.
        # So if the error is big, it detracts from log probability
        # more than if the error is small.
        return -l2_err

    def find_best_sequence(self):
        """Uses a Viterbi-style algorithm to decode the best sequence through the \
        chord-map of the piece, returning which chord is predicted to be sounding \
        at each frame of the transformed audio data. \

        Uses the idea from section 10.4 of the Viberbi Algorithm to \
        find the most likely sequence.

        Returns:
            best_sequence ((self.num_samples_) ndarray): The best sequence 
                through the same-shaped transformed audio samples.
        """ 
        # Initialize a giant matrix of bellman optimality stuff
        num_chords_in_sequence = self.A.shape[0]
        optimal_matrix = np.zeros((self.num_samples, num_chords_in_sequence))
        optimizer_matrix = np.zeros((self.num_samples - 1, num_chords_in_sequence), dtype=np.int32)

        # Initialize a vector of appropriate log-probabilities 
        pi = np.repeat(-np.inf, num_chords_in_sequence)       # Initial state distribution 
        pi[0] = 0               # [must start at first silence state - everything else
                                # has probability zero, or log probability -infinity]
        eta_0 = pi + self._find_spectrum_log_prob(self.audio_events[0])
        optimal_matrix[0] = eta_0 
        
        # Iterate the forward portion of the Viterbi algorithm (finding all of the etas,
        # and populating them into the lookup matrix optimal_matrix), using log-probabilities 
        # (Instead of regular b's, we use _find_spectrum_prob() )
        print("Populating Bellman log-probabilities . . .")
        with np.errstate(divide='ignore'):  # Ignores trying to take log of zero
            log_A = np.log(self.A)      # Log-probability of transition matrix
            for t in range(1, self.num_samples):
                print(f"Iteration {t}/{self.num_samples}", end='\r')
                eta_prev = optimal_matrix[t-1] 
                b_t = self._find_spectrum_log_prob(self.audio_events[t])
                big_matrix = eta_prev + log_A + b_t[:,np.newaxis]
                optimal_matrix[t] = np.max(big_matrix, axis=1)
                optimizer_matrix[t-1] = np.argmax(big_matrix, axis=1) 
        
        # Store these as class attributes 
        self.optimal_matrix = optimal_matrix 
        self.optimizer_matrix = optimizer_matrix

        # Now, iterate the backward portion of the Viterbi algorithm 
        # to construct the state from end to beginning 
        print("Populating best sequence . . .")
        best_sequence = np.zeros(self.num_samples, dtype=np.int32) 
        best_sequence[-1] = np.argmax(optimal_matrix[-1])
        for i in range(2,self.num_samples+1):
            best_sequence[-i] = optimizer_matrix[-i+1,best_sequence[-i+1]]
        
        # Return the best sequence indexes 
        print("Done.")
        return best_sequence 
    







