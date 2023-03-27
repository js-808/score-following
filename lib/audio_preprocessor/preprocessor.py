### A library used to preprocess audio data 
from typing import *
import py_midicsv as pm     # Used for MIDI preprocessing
from multiset import Multiset
import pandas as pd 
import os 
import librosa
from pydub import AudioSegment
import re
from scipy.stats import norm
from pydub import AudioSegment

class InvalidMIDIFileException(Exception):
    """An exception that is raised when a MIDI file is invalid
    and cannot be parsed by the AudioPreprocessor"""
    pass 

class AudioPreprocessor: 
    """A class to preprocess audio and MIDI data to use in a 
    score following algorithm
    
    Attributes:
        NOTES_TRACK_NUMBER (int): The track number in the MIDI files \
            for the notes we'd like to transcribe
        NOTE_EVENT_NAMES (List[str]): A list of MIDI event names \
            corresponding to played notes on the piano
        midi_file (str): A fully qualified filepath to the MIDI file \
            of the piece we are processing 
        csv_file (str): A fully qualified filepath to the CSV file \
            parsed from the MIDI file
    """
    ### CONSTANT ATTRIBUTES 
    NOTES_TRACK_NUMBER : int = 2 
    MIDI_PITCH_COLUMN : int = 'event_value2'
    ONSET_EVENT_NAME : str = 'Note_on_c'
    OFFSET_EVENT_NAME : str = 'Note_off_c'
    NOTE_EVENT_NAMES : List[str] = [ONSET_EVENT_NAME, OFFSET_EVENT_NAME]

    ### METHODS
    def __init__(self, midi_file : str, wav_file : str):
        # Parse the MIDI to CSV, and store it in the parsed_csvs directory
        self.midi_file : str = midi_file 
        self.csv_file : str = self._midi_to_csv(midi_file)
        self.wav_file : str = wav_file
        self.chord_events : List[Tuple] = self._parse_csv(self.csv_file)

    def _midi_to_csv(self, midi_file : str) -> str:
        """Parse a MIDI file to a CSV file in the same folder 
        
        Parameters:
            midi_file (str): A fully qualified filepath to the MIDI file \
                to parse 
        Returns:
            csv_file (str): A fully qualified filepath to the parsed \
                CSV file
        """
        # If the folder 'parsed_csvs' does not exist in the same folder
        # as the MIDI file, create it 
        midi_parent_dir = os.path.dirname(midi_file)
        csv_dir = os.path.join(midi_parent_dir, 'parsed_csvs')
        if not os.path.isdir(csv_dir):
            os.mkdir(csv_dir)

        # Get a full path to the output CSV file (inside a directory called 'parsed_csvs')
        midi_filename_no_ext = os.path.splitext(os.path.basename(midi_file))[0]
        csv_filename = midi_filename_no_ext + '.csv'
        csv_file = os.path.join(csv_dir, csv_filename)

        # Load the MIDI file and parse it into CSV format
        csv_string = pm.midi_to_csv(midi_file)

        # Write the CSV out to the desired output location
        with open(csv_file, "w") as f:
            f.writelines(csv_string) 

        return csv_file 

    def _parse_csv(self, csv_path : str):
        """Parses one of the MIDI CSVs into a sequence of note values.
        
        Parameters:
            csv_path (str): A fully qualified filepath to the parsed \
                CSV file 
        """
        # Get the CSV in the format specified online
        col_names = ['track', 'time', 'event_name', 'event_value1', 'event_value2', 'event_value3', 'event_value4']
        piece = pd.read_csv(csv_path, names=col_names, sep=', ', engine='python')

        # Extract only the notes from the MIDI (ignore all extra info)
        piece = piece.loc[piece['track']==self.NOTES_TRACK_NUMBER]
        piece = piece.loc[piece['event_name'].isin(self.NOTE_EVENT_NAMES)]
        piece[self.MIDI_PITCH_COLUMN] = pd.to_numeric(piece[self.MIDI_PITCH_COLUMN])

        # Get only the columns we need to process note info 
        piece = piece[['time', 'event_name', self.MIDI_PITCH_COLUMN]]
        piece.rename(columns = {'event_value2': 'MIDI_pitch_num'}, inplace=True)

        # Get individual note events as a list of tuples, sorted by 
        # onset times 
        note_events = self._get_note_info(piece)

        # Get chord events as a list of tuples, and return them 
        chord_events = self._get_chord_events(note_events)
        return chord_events 

    def _get_note_info(self, piece : pd.DataFrame) -> List[Tuple]:
        """Parse note information from a DataFrame representation of the note
        events in a MIDI file.
        
        Parameters:
            piece (pd.Dataframe): A DataFrame containing columns \
                [time, event_name, MIDI_pitch_num] 
        
        Returns:
            note_events (List[Tuple]): A list with entries \
                [(time, '+' (for onset) or '-' (for offset), MIDI Note Value)]
        Raises:
            InvalidMIDIFileException : If the number of onset events \
                does not equal the number of offset events 
        """
        # Parse the onset and offset events
        on_events = piece.loc[piece['event_name'] == self.ONSET_EVENT_NAME]
        off_events = piece.loc[piece['event_name'] == self.OFFSET_EVENT_NAME]

        # Double check we have the same number of events. If not, terminate early
        if len(on_events) != len(off_events):
            msg = f"Error: {self.midi_file} has unequal number of note onsets and note offsets"
            raise InvalidMIDIFileException(msg)

        # Parse note onsets and offsets as tuples 
        on_events_tups = [(tup[0], '+', tup[1]) for tup in zip(on_events['time'], on_events['MIDI_pitch_num'])] 
        off_events_tups = [(tup[0], '-', tup[1]) for tup in zip(off_events['time'], off_events['MIDI_pitch_num'])]

        # Combine these two lists, and sort by time (and secondarily by '-' before '+', and lastly, by pitch number)
        note_events = on_events_tups + off_events_tups 
        note_events = sorted(note_events, key=lambda x:(-x[0], x[1], -x[2]), reverse=True)

        return note_events 

    def _get_chord_events(self, note_events : Sequence[Tuple]) -> List[Tuple]:
        """Get chords from a sequence of note events.
        Inspired by 
        https://stackoverflow.com/questions/628837/how-to-divide-a-set-of-overlapping-ranges-into-non-overlapping-ranges
        Parameters:
            note_events (List[Tuple]): A list with entries \
                [(time, '+' (for onset) or '-' (for offset), MIDI Note Value)]
        Returns:
            chord_events (List[Tuple]): A list with entries \
                [([List of chord note_values], onset_time, offset_time)]
                (intervals should be non-intersecting, and should span 
                the entire time interval (with any silence being represented
                by an empty list on that time interval))
        """
        chord_events = []            # A list of all interval chord events (onset, offset, {notes in chord})
        currently_sounding = Multiset()   # Set of MIDI note values currently sounding
        last_event_time = 0          # Used for subdividing intervals 
        for cur_time, marker, midi_val in note_events:
            # If we have moved our current time, record which notes were sounding
            # from the previous event time to the current event time
            if cur_time != last_event_time: # Current time splits into old interval
                chord_events.append((last_event_time, cur_time, sorted(list(set(currently_sounding)))))

            # Add/remove notes from currently sounding based off of what kind of event this is 
            if marker == '+':       # Onset event 
                currently_sounding.add(midi_val)
            elif marker == '-':     # Offset event 
                currently_sounding.discard(midi_val, 1)

            # Update the event time 
            last_event_time = cur_time 

        # Return our chord events 
        return chord_events 

    def continuous_q(self, wav_path, n_bins=88):
        """Gets the continuous Q transform for audio file
        Parameters
            wav_path (str): Filepath to wav file
            n_bins (int): Number of frequency bins
        
        Returns
            cqts (ndarray(n_bins, n_frames)): list of continuous Q transforms"""
        y, sr = librosa.load(wav_path)
        cqt = (librosa.cqt(y, sr=sr, n_bins=n_bins))

        return cqt

    def get_mfcc(self, wav_path, delta_bool = False):
        """Gets the MFCC coefficients for an audio file as well as the \
            estimate of the derivative if desired
        
        Parameters
            wav_path (list): List of filepaths to wav file
            delta_bool (bool): Boolean for whether or not we want to compute \
                the delta feature of the wav file
            
        Returns 
            mfcc_coeffs (ndarray(n_mfcc, n_frames)): MFCC coefficients for \
                each file
            mfcc_delta_coeffs (ndarray(n_mfcc, n_frames)): Local estimate of \
                the derivative of the input file"""
        y, sr = librosa.load(wav_path)
        mfcc_coeffs = librosa.feature.mfcc(y=y, sr=sr)

        if delta_bool:
            mfcc_delta_coeffs = librosa.feature.delta(mfcc_coeffs)
            return mfcc_coeffs, mfcc_delta_coeffs

        return mfcc_coeffs

    def misc_data(self, wav_path):
        """Gets miscellaneous data from a .wav file including estimated \
            tempo, beat frames, beat times
        
        Parameters
            filepaths: list of filepaths
            
        Returns 
            tempo (float64): Estimated tempo for each file
            beat_frame (ndarray()): Beat frames for each file
            beat_time (ndarray()): Beat times for each file"""

        y, sr = librosa.load(wav_path)
        tempo, beat_frame = librosa.beat.beat_track(y=y, sr=sr)
        beat_time = librosa.frames_to_time(beat_frame, sr=sr)

        return tempo, beat_frame, beat_time


    # Reset the file path for the wav files
    if fugue.search(ogg_file_path):        
        wav_file_path = '../../data/wav_files/wtk1-fugue' + number + '.wav'
    elif prelude.search(ogg_file_path):
        wav_file_path = '../../data/wav_files/wtk1-prelude' + number + '.wav'
    
    # Convert ogg files to wav files
    x = AudioSegment.from_file(ogg_file_path)
    x.export(wav_file_path, format='wav')
    return wav_file_path
def mp3_to_wav_conversion(mp3_file):
    """Converts MP3 to WAV file in the same directory.
    
    Parameters:
        mp3_file (str): A fully qualified path to the MP3 file to convert 
    
    Returns:
        wav_file (str): A fully qualified path to the newly created WAV file
    """
    # Get output path to WAV file to export                                                                 
    wav_file = os.path.splitext(mp3_file)[0] + '.wav'

    # Convert wav to mp3                                                            
    sound = AudioSegment.from_mp3(mp3_file)
    sound.export(wav_file, format="wav")

if __name__ == "__main__":
    midi_file = '../../data/midi/wtk1-fugue8.mid'
    wav_file = ogg_to_wav('../../data/ogg_files/Kimiko_Ishizaka_-_Bach_-_Well-Tempered_Clavier,_Book_1_-_16_Fugue_No._8_in_D-sharp_minor,_BWV_853.ogg')

    ap = AudioPreprocessor(midi_file, wav_file)   
    cqt = abs(ap.continuous_q(wav_file))
    