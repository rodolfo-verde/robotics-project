import speech_recognition as sr  
import pyttsx3
import time

# Initialize the recognizer
r = sr.Recognizer()

# Function to convert text to
# speech
def SpeakText(command):
    
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()

# Use the microphone as source for input.
with sr.Microphone() as source2:
    # wait for a second to let the recognizer
    # adjust the energy threshold based on
    # the surrounding noise level
    r.adjust_for_ambient_noise(source2, duration=0.2)
    print("Say Something")
    #listens for the user's input
    audio2 = r.listen(source2)

# Using google to recognize audio
MyText = r.recognize_google(audio2)
MyText = MyText.lower()

print("Did you say "+MyText)
SpeakText(MyText)


"""for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print('Microphone with name "{1}" found for `Microphone(device_index={0})`'.format(index, name))
"""
"""# get audio from the microphone                                                                       
r = sr.Recognizer()                                                                                   
with sr.Microphone(device_index=1) as source:                                                                       
    print("Speak:")
    starttime = time.time()                                                                                   
    audio = r.listen(source)   
    endtime = time.time()
    print(f"Time needed for listening: {endtime-starttime}")
    print("Starting to recognize...")
    starttime = time.time()
    print("You said " + r.recognize_google(audio))
    print(f"Time needed: {time.time()-starttime}")"""

"""try:
    print("Starting to recognize...")
    starttime = time.time()
    print("You said " + r.recognize_google(audio))
    print(f"Time needed: {time.time()-starttime}")
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))"""