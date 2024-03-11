import os
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D
from keras.optimizers import SGD
from keras.regularizers import L2 
from keras.models import load_model
import time
import sounddevice as sd

from audio_processing.dataprocessor import dataprocessor
from audio_processing.mfcc_processor import mfcc_dataprocessor

from robot_control.TickTackToe import TickTackToe

from audio_processing.word_logic import WordLogic
from audio_processing.AudioStream import AudioStream

from scipy.io import wavfile
import I2C_LCD_driver

import queue
import threading

# import big speech model
#model.load_weights("audio_processing//CNN_Models//CNN_Model_Rex_simple.h5")
# Load the saved model
print("Loading model...")

try:
    model = tf.keras.models.load_model('audio_processing//CNN_Models//CNN_Model_Rex_simple.h5')
    model.summary()
except Exception as e:
    print(e)
    quit()

print("Model loaded.")

#safe1 stores the input from the stream to be processed later
        
safe1 = np.array([], dtype="float64")


# This is the Network class
# It will be used to run the game

class Network:
    def __init__(self):
        # this is the constructor of the class
        # it will initialize the CNN model and the TickTackToe game
        self.audio_stream = AudioStream()
        self.class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]
        self.mylcd = I2C_LCD_driver.lcd()
        self.mylcd.lcd_clear()
        self.wordlogic = WordLogic(self.mylcd)
        self.ticktacktoe = TickTackToe(solo_play=False, display = None)

    def run(self):
        try:                
            # this runs the game
            def check_for_stop_command():
                while True:
                    try:
                        # Try to get a command from the stop_command_queue without blocking
                        command = self.audio_stream.stopp_command_queue.get_nowait()
                        if command == "Stopp":
                            print("Stopping the robot...")
                            self.ticktacktoe.command("Stopp")
                    except queue.Empty:
                        time.sleep(0.1)
            
            # Start the thread that checks for the "Stopp" command
            stop_thread = threading.Thread(target=check_for_stop_command, daemon=True)
            stop_thread.start()
            while True:
                try:
                    command = self.audio_queue.get_nowait()
                    print(f"Command from audio_queue: {command}")
                    self.wordlogic.command(command)
                    if self.wordlogic.get_combination() not in ["", None]:
                        self.ticktacktoe.command(self.wordlogic.get_combination())
                        while self.ticktacktoe.playing:
                            self.mylcd.lcd_display_string(f"REX     {self.wordlogic.get_combination().upper()}     %",2)
                            time.sleep(.21)
                            self.mylcd.lcd_display_string(f"REX     {self.wordlogic.get_combination().upper()}      ",2)
                            time.sleep(.21)                
                        self.wordlogic.reset_combination()
                    elif command == "Other":
                        pass
                except queue.Empty:
                    time.sleep(0.1)
                    continue
        except KeyboardInterrupt:
            print("Interrupted by user...")
        except Exception as e:
            print(f"Error: {e}")

    
    def start_audio_stream(self):
        self.audio_stream_thread = threading.Thread(target=self.audio_stream.main, daemon=True)
        self.audio_stream_thread.start()

                    
if __name__ == '__main__':
    network = Network()
    network.start_audio_stream()
    network.run()
    
    
