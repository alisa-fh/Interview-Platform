from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Activation, BatchNormalization, Dropout, MaxPooling1D, Flatten, Dense
import librosa
import numpy as np
import pandas as pd
import pyaudio
from array import array
from sys import byteorder
import wave
from struct import pack
from keras.models import model_from_json
import time
input_duration = 3
THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def record():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    start = time.time()
    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

        if time.time() - start >=5:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

# Model
class Model:
    def __init__(self, num_classes=2): #num_classes can be 2, 10
        self.num_classes = num_classes
    def getModel(self):
        if self.num_classes == 2:
            json_file = open('./model-2cat/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights("./model-2cat/weights.h5")
        elif self.num_classes == 8:
            json_file = open('./model-8cat/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights("./model-8cat/weights.h5")
        else:
            raise ValueError('num_classes must be 2 or 5')
        return model


# model = Model(num_classes=2)
# model = model.getModel()
# path_array = ["audio-files/actor21.wav", "audio-files/actor21-1.wav", "audio-files/actor21-2.wav", "audio-files/actor21-3.wav"]
# for audio_path in path_array:
#     X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=input_duration, sr=22050 * 2,
#                                       offset=0.5)
#     sample_rate = np.array(sample_rate)
#     mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
#     feature = mfccs
#     data_test = pd.DataFrame(columns=['feature'])
#     data_test.loc[0] = [feature]
#
#     test_valid = pd.DataFrame(data_test['feature'].values.tolist())
#     test_valid = np.array(test_valid)
#     test_valid = np.expand_dims(test_valid, axis=2)
#     preds = model.predict(test_valid,
#                              verbose=1)
#     # negative, positive
#     if np.argmax(preds) == 0:
#         print("negative")
#     elif np.argmax(preds) == 1:
#         print("positive")
#     else:
#         print("error")
#     print(audio_path)
#     print(preds)

print("Please talk")
audio_path = "test.wav"
record_to_file(audio_path)

model = Model(num_classes=2)
model = model.getModel()

X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=input_duration, sr=22050 * 2,
                                  offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
feature = mfccs
data_test = pd.DataFrame(columns=['feature'])
data_test.loc[0] = [feature]

test_valid = pd.DataFrame(data_test['feature'].values.tolist())
test_valid = np.array(test_valid)
test_valid = np.expand_dims(test_valid, axis=2)
preds = model.predict(test_valid,
                         verbose=1)
# negative, positive
if np.argmax(preds) == 0:
    print("negative")
elif np.argmax(preds) == 1:
    print("positive")
else:
    print("error")
print(audio_path)
print(preds)