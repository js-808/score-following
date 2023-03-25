### A library used to preprocess audio data 
from typing import *
import py_midicsv as pm     # Used for MIDI preprocessing
import pandas as pd 
import os 

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
    NOTE_EVENT_NAMES : List[str] = ['Note_on_c', 'Note_off_c']
    MIDI_PITCH_COLUMN : int = 'event_value2'

    ### METHODS
    def __init__(self, midi_file : str):
        # Parse the MIDI to CSV, and store it in the parsed_csvs directory
        self.midi_file : str = midi_file 
        self.csv_file : str = self._midi_to_csv(midi_file)

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

    def _parse_csv(self, csv_path : str):
        """Parses one of the MIDI CSVs into a sequence of note values"""
        # Get the CSV in the format specified online
        col_names = ['track', 'time', 'event_name', 'event_value1', 'event_value2', 'event_value3', 'event_value4']
        piece = pd.read_csv(csv_path, names=col_names, sep=', ', engine='python')

        # Extract only the notes from the MIDI (ignore all extra info)
        piece = piece.loc[piece['track']==self.NOTES_TRACK_NUMBER]
        piece = piece.loc[piece['event_name'].isin(self.NOTE_EVENT_NAMES)]

        # 

                