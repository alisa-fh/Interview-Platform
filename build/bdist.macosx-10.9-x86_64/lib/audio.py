import pyaudio
from array import array
from sys import byteorder
import wave
from struct import pack
import time

from utils import extract_feature
import pickle

audio_feedback_q = 0

input_duration = 3
THRESHOLD = 500
CHUNK_SIZE = 2048 #1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30

killAudioThread = False
negative_result = ""
result = ""
post_results = []

def getNegativeResult(neg_result):
    global negative_result
    global result
    if neg_result:
        return negative_result
    else:
        if len(post_results) < 3:
            return []
        else:
            return post_results

def clearNegativeResult():
    post_results = []

def clearAudioFeedbackQueue():
    global audio_feedback_q
    prev = audio_feedback_q
    audio_feedback_q = 0
    return prev

def getAudioFeedbackQueue():
    return audio_feedback_q


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
    try:
        times = float(MAXIMUM)/max(abs(i) for i in snd_data)
    except:
        return -1 # if error

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def snd(stream, q):
    snd_data = array('h', stream.read(CHUNK_SIZE))
    q.put(snd_data)



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
    global killAudioThread
    print("in record")
    while 1:
        print("time: ", time.time() - start, "kill audio ", killAudioThread)
        # little endian, signed short
        # q = multiprocessing.Queue()
        # p = multiprocessing.Process(target=snd, args=(stream,q))
        # p.start()
        # p.join(2)
        # if p.is_alive():
        #     print("killing process")
        #     p.terminate()
        #     # OR Kill - will work for sure, no chance for process to finish nicely however
        #     # p.kill()
        #     p.join()
        #     killAudioThread = True
        #     r = -2
        #     break
        # snd_data = q.get()

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

        if  killAudioThread: #time.time() - start >= 5 or
            break

    if r == -2:
        return 0, 0, 2
    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    if r == -1: # if error
        return sample_width, -1, 2
    r = trim(r)
    r = add_silence(r, 0.5)
    if killAudioThread == True:
        return sample_width, r, 1

    return sample_width, r, 0

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    print("in record to file")
    sample_width, data, exit = record()
    if exit == 2:
        return exit
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    # exit is 1 for exit, 0 for no exit, 2 for error
    return exit

# Second code

def changeKillAudioThread(val):
    global killAudioThread
    killAudioThread = val

def initModel():
    global audiomodel
    audiomodel = pickle.load(open("mlp_classifier.model", "rb"))



def audioAnalysis():
    global killAudioThread
    global audio_feedback_q
    global negative_result
    global result
    global post_results
    print("in audioanalysis")
    isAnsweringQuestion = True
    negative_clips = 0
    currentlyFeedbacking = False

    while isAnsweringQuestion:
        print("Please talk")
        audio_path = "test.wav"
        exit = record_to_file(audio_path)
        print("killaudiothread",killAudioThread)
        if exit == 1 or killAudioThread == True:
            print("audioanalysis: killAudioThread true, should break")
            isAnsweringQuestion = False
            negative_clips = 0
            break
        if exit == 2: #if error
            print("there was an error. continuing")
            continue
        features = extract_feature(audio_path, mfcc=True, chroma=True, mel=True).reshape(1, -1)
        # predict
        result = audiomodel.predict(features)[0]
        # negative, positive
        if result in ['angry', 'sad']:
            print("negative")
            negative_clips += 1
            print("negative voice")
        elif result in ['neutral', 'happy']:
            print("positive")
            negative_clips += 1
            print("positive voice")
        else:
            print(result)
        print("negative clips: ", negative_clips)
        if currentlyFeedbacking and len(post_results) < 3:
            post_results.append(result)
            if len(post_results) == 3:
                currentlyFeedbacking = False

        if negative_clips >= 3:
            negative_clips = 0
            print("3 negative clips")
            audio_feedback_q += 1
            negative_result = result
            currentlyFeedbacking = True


