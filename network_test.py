print("test")

import time

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
        self.ticktacktoe = None
        self.round = 0
        self.game_type = False

    def run(self):            
        try:
            while self.ticktacktoe == None:
                try:                    
                    self.mylcd.lcd_display_string(f"1 = Alleine!",1)
                    self.mylcd.lcd_display_string(f"2 = Zu zweit!",2)
                    command = self.audio_stream.command_queue.get_nowait()
                    if command == "1":
                        command = None
                        print("I heared 1")
                        self.ticktacktoe = TickTackToe(solo_play=True, start = 0, display = None)
                        #print("I got even further")
                        self.mylcd.lcd_clear()
                        self.game_type = True
                    elif command == "2":
                        command = None
                        print("I heared 2")
                        self.ticktacktoe = TickTackToe(solo_play=False, display = None)
                        self.mylcd.lcd_clear()
                        self.game_type = True
                    else:
                        continue
                except queue.Empty:
                    time.sleep(0.1)                
            # this runs the game
            def check_for_stop_command():
                while True:
                    try:
                        # Try to get a command from the stop_command_queue without blocking
                        command = self.audio_stream.stopp_command_queue.get_nowait()
                        if command == "stopp":
                            print("Stopping the robot...")
                            self.ticktacktoe.command("Stopp")
                                                    
                    except queue.Empty:
                        time.sleep(0.1)
            
            # Start the thread that checks for the "Stopp" command
            stop_thread = threading.Thread(target=check_for_stop_command, daemon=True)
            stop_thread.start()
            while self.game_type == True:
                try:
                    if not self.ticktacktoe._game_over:
                        self.switch_player()                        
                        command = self.audio_stream.command_queue.get_nowait()
                        #print(f"Command from audio_queue: {command}")
                        self.wordlogic.command(command)
                        if self.ticktacktoe.rex_stopp:
                            print("im here")                               
                            while command != "rex":                                    
                                try:
                                    command = self.audio_stream.command_queue.get_nowait()
                                    print(f'Got {command}')
                                except queue.Empty:
                                    command = ""
                                time.sleep(0.1)
                            print("im here aswell")
                            self.mylcd.lcd_display_string(f"REX          %",2)
                            self.ticktacktoe.rex_play()
                            self.ticktacktoe._rex_stopp = False                       
                        if self.wordlogic.get_combination() not in ["", None]:
                            self.ticktacktoe.command(self.wordlogic.get_combination())
                            while self.ticktacktoe.playing:
                                self.mylcd.lcd_display_string(f"REX     {self.wordlogic.get_combination().upper()}     %",2)
                                time.sleep(.21)
                                self.mylcd.lcd_display_string(f"REX     {self.wordlogic.get_combination().upper()}      ",2)
                                time.sleep(.21)
                            self.wordlogic.reset_combination()
                            if self.ticktacktoe.cmd == "Stopp":
                                self.mylcd.lcd_display_string(f"REX     Stoppt!     %",2)
                                time.sleep(2)                
                            self.mylcd.lcd_clear()
                        elif command == "Other":
                            pass
                    else:
                        print(self.ticktacktoe._check_draw())
                        if self.ticktacktoe._check_draw():
                            self.mylcd.lcd_clear()
                            self.mylcd.lcd_display_string(f"Unentschieden!",1)
                            self.mylcd.lcd_display_string(f"Rex raeumt auf!",2)
                            self.game_type = False
                            #self.ticktacktoe = None
                        elif np.abs(self.ticktacktoe._current_player-1) == 0:
                            self.mylcd.lcd_clear()
                            self.mylcd.lcd_display_string(f"Rot gewinnt!",1)
                            self.mylcd.lcd_display_string(f"Rex raeumt auf!",2)
                            self.game_type = False
                            #self.ticktacktoe = None
                        else:
                            self.mylcd.lcd_clear()
                            self.mylcd.lcd_display_string(f"Blau gewinnt!",1)
                            self.mylcd.lcd_display_string(f"Rex raeumt auf!",2)
                            self.game_type = False
                            #self.ticktacktoe = None
                                                                                  
                except queue.Empty:
                    time.sleep(0.1)
                    continue
            #print("hekko")
            while self.ticktacktoe._resetplay == 1:
                time.sleep(0.1)
                #print("cleaning")                
            else:
                #print("dead")
                self.ticktacktoe = None
                self.mylcd.lcd_clear()
                self.run()
        except KeyboardInterrupt:
            print("Interrupted by user...")
        except Exception as e:
            print(f"Error: {e}")

    
    def start_audio_stream(self):
        self.audio_stream_thread = threading.Thread(target=self.audio_stream.main, daemon=True)
        self.audio_stream_thread.start()
    
    def switch_player(self):
        if self.ticktacktoe.rex_playing:
            self.mylcd.lcd_display_string(f"Rex spielt!",1)
        elif self.ticktacktoe._current_player == 0:
            #self.ticktacktoe_rex_playing=False
            self.mylcd.lcd_display_string(f"Rot Spielt!",1)
        else:
            #self.ticktacktoe_rex_playing=False
            self.mylcd.lcd_display_string(f"Blau Spielt!",1)
            
        

                    
if __name__ == '__main__':
    network = Network()
    network.start_audio_stream()
    network.run()
    
    
    


    


