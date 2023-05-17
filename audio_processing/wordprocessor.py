import numpy as np
import sounddevice as sd

class wordprocessor:

    SAMPLERATE: int
    outi = sd.OutputStream

    def __init__(self, samplerate: int) -> None:
        self.SAMPLERATE = samplerate
        self.outi = sd.OutputStream(self.SAMPLERATE, device=6)

    
    def playsound(self, data: np.array):

        print(data.shape)

        sd.play(data, self.SAMPLERATE)