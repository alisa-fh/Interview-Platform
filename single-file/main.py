#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import math
import pandas as pd
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

from threading import Thread

import sys


########### UTILS.py #############
import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.preprocessing import label
from sklearn.model_selection import train_test_split
from sklearn import neural_network
from sklearn.neural_network import multilayer_perceptron


# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# we allow only these emotions
AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy"
}

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("data/Actor_*/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

############# audio.py ##############
import pyaudio
from array import array
from sys import byteorder
import wave
from struct import pack
import time
import pickle

audio_audio_feedback_q = 0

input_duration = 3
THRESHOLD = 500
CHUNK_SIZE = 2048 #1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30

audiokillAudioThread = False
audionegative_result = ""
audioresult = ""
audiopost_results = []

def getNegativeResult(neg_result):
    global audionegative_result
    global audioresult
    if neg_result:
        return audionegative_result
    else:
        if len(audiopost_results) < 3:
            return []
        else:
            return audiopost_results

def clearNegativeResult():
    global audiopost_results
    audioaudiopost_results = []

def clearAudioFeedbackQueue():
    global audio_audio_feedback_q
    prev = audio_audio_feedback_q
    audio_audio_feedback_q = 0
    return prev

def getAudioFeedbackQueue():
    return audio_audio_feedback_q


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
    global audiokillAudioThread
    print("in record")
    while 1:
        print("time: ", time.time() - start, "kill audio ", audiokillAudioThread)
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

        if  audiokillAudioThread: #time.time() - start >= 5 or
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
    if audiokillAudioThread == True:
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
    global audiokillAudioThread
    audiokillAudioThread = val

def initModel():
    global audiomodel
    bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    path_to_model = os.path.abspath(os.path.join(bundle_dir, 'mlp_classifier.model'))

    audiomodel = pickle.load(open(path_to_model, "rb"))



def audioAnalysis():
    global audiokillAudioThread
    global audio_audio_feedback_q
    global audionegative_result
    global audioresult
    global audiopost_results
    print("in audioanalysis")
    isAnsweringQuestion = True
    negative_clips = 0
    audiocurrentlyFeedbacking = False

    while isAnsweringQuestion:
        print("Please talk")
        audio_path = "test.wav"
        exit = record_to_file(audio_path)
        print("killaudiothread",audiokillAudioThread)
        if exit == 1 or audiokillAudioThread == True:
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
        if audiocurrentlyFeedbacking and len(audiopost_results) < 3:
            audiopost_results.append(result)
            if len(audiopost_results) == 3:
                audiocurrentlyFeedbacking = False

        if negative_clips >= 3:
            negative_clips = 0
            print("3 negative clips")
            audio_audio_feedback_q += 1
            audionegative_result = result
            audiocurrentlyFeedbacking = True





# Initialise audio model
initModel()

# Facial odel used
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
path_to_model = os.path.abspath(os.path.join(bundle_dir, 'model.h5'))

# Load pre-trained weights
model.load_weights(path_to_model)
# Load facial detection

bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
path_to_face = os.path.abspath(os.path.join(bundle_dir, 'haarcascade_frontalface_default.xml'))

face_haar_cascade = cv2.CascadeClassifier(path_to_face)

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
negative_indices = [0, 1, 2, 5]
positive_indices = [3, 4]

# Results
pre_feedback = []
post_feedback = []
feedback_type = []
question_list = []

# Feedback
bald = [ "Consider each potential word definition carefully",
         "Ensure you have read the entire text slowly.",
         "Consider which word(s) make the text flow best.",
         "Make sure the whole text is understood.",
         "Consider which words sound best in the context.",
         "Consider which words are most similar.",
         "Think about one word at a time."
]

positive_politeness = [ "We can do this, why don’t we consider each word’s definition?",
                        "Gettting there, let’s make sure we’ve read through the text slowly!",
                        "Shall we see which words seem most similar?",
                        "Let’s see if we can figure which words fit best.",
                        "Shall we take it one word at a time?",
                        "Good concentration! We can try eliminating unlikely words."
]

feedback_dictionary = {0: "", 1: bald, 2: positive_politeness}



def main():
    # Read question CSV into data frame
    qu_df = readQuestionFile()
    print(qu_df)
    question_number = 1
    sg.theme('DefaultNoMoreNagging')

    task, question, options, answers, actual_difficulty, label_difficulty, time_length = getNextQuestion(qu_df, question_number)
    options = options.split(', ')
    user_answers = []
    question_number = 1

    if label_difficulty == 'Hard':
        difficulty_color = '#d12e30'
    else:
        difficulty_color = '#33cc4f'

    left_checkboxes = [[sg.Text('(i)', font='Helvetica 17', justification="centre", key="i", size=(5,1), visible=False),
                    sg.Checkbox('Option 1', font='Helvetica 18', key='C1', size=(15,1), visible = False),
                   sg.Checkbox('Option 2', font='Helvetica 18', key='C2',size=(15,1), visible = False),
                   sg.Checkbox('Option 3', font='Helvetica 18', key='C3',size=(15,1), visible = False)]]
    right_checkboxes = [[sg.Text('(ii)', font='Helvetica 17', justification="centre", key="ii", size=(5,1), visible=False),
                    sg.Checkbox('Option 4', font='Helvetica 18', key='C4',size=(15,1), visible = False),
                   sg.Checkbox('Option 5', font='Helvetica 18', key='C5',size=(15,1), visible = False),
                   sg.Checkbox('Option 6', font='Helvetica 18', key='C6',size=(15,1), visible = False)]]

    left_column = [[sg.Text('Task:\n ' + task, size=(50, 5), justification='center', font='Helvetica 20', key='task',
                            background_color='#c1c1c1')],

                   [sg.Text()],

                   [sg.Text('Question ' + str(question_number), size=(8, 1), font='Helvetica 20', key='question_number'),
                    sg.Text(label_difficulty, size=(4, 1), font='Helvetica 20', justification= "left", key='difficulty',
                            background_color= difficulty_color),
                    sg.Text('Given time: ' + time.strftime("%H:%M:%S", time.gmtime(time_length))[3:], size=(15, 1),
                            font='Helvetica 20', justification="left", key='time')
                    ],

                   [sg.Text(question, size=(50, 8),
                            font='Helvetica 20', justification="left", key='qu', visible=False),
                    sg.Text()],

                    [sg.Column(left_checkboxes),
                    sg.Column(right_checkboxes)]

                   ]


    right_column = [[sg.Image(filename='', key='image')],
                    [sg.Text("\n\n\n", size=(50, 1),
                             font='Helvetica 20', text_color="red", justification="left", key='feedback', visible=True)]
                    ]

    layout = [
        [
        sg.Column(left_column),
        sg.VSeperator(),
        sg.Column(right_column),
        ],
        [sg.Button('Start', size=(10, 1), font='Helvetica 14'),
        sg.Button('Submit', size=(10, 1), font='Any 14', disabled=True),
        sg.Button('Next Question', size=(15, 1), font='Helvetica 14', visible=False),
        sg.Button('Exit', size=(10, 1), font='Helvetica 14', button_color="red")]
        ]

    questionno_column = [[sg.Text("Question:", size=(9, 3), font='Helvetica 14', justification="left")],
                        [sg.Text("1", size=(2, 5), font='Helvetica 14', justification="left")],
                    [sg.Text("2", size=(2, 5), font='Helvetica 14', justification="left")],
                       [sg.Text("3", size=(2, 5), font='Helvetica 14', justification="left")],
                       [sg.Text("4", size=(2, 5), font='Helvetica 14', justification="left")],
                       [sg.Text("5", size=(2, 5), font='Helvetica 14', justification="left")],
                       [sg.Text("6", size=(2, 5), font='Helvetica 14', justification="left")]]
    givans_column = [[sg.Text("You answered:", size=(13, 3), font='Helvetica 14', justification="left")],
                    [sg.Text("1", size=(13, 5), font='Helvetica 14', justification="left", key="givans1")],
                    [sg.Text("2", size=(13, 5), font='Helvetica 14', justification="left", key="givans2")],
                       [sg.Text("3", size=(13, 5), font='Helvetica 14', justification="left", key="givans3")],
                       [sg.Text("4", size=(13, 5), font='Helvetica 14', justification="left", key="givans4")],
                       [sg.Text("5", size=(13, 5), font='Helvetica 14', justification="left", key="givans5")],
                       [sg.Text("6", size=(13, 5), font='Helvetica 14', justification="left", key="givans6")]]
    actans_column = [[sg.Text("Correct answer:", size=(15, 3), font='Helvetica 14', justification="left")],
                    [sg.Text("1", size=(10, 5), font='Helvetica 14', justification="left", key="actans1")],
                     [sg.Text("2", size=(10, 5), font='Helvetica 14', justification="left", key="actans2")],
                     [sg.Text("3", size=(10, 5), font='Helvetica 14', justification="left", key="actans3")],
                     [sg.Text("4", size=(10, 5), font='Helvetica 14', justification="left", key="actans4")],
                     [sg.Text("5", size=(10, 5), font='Helvetica 14', justification="left", key="actans5")],
                     [sg.Text("6", size=(10, 5), font='Helvetica 14', justification="left", key="actans6")]]

    answer_layout = [
        [sg.Column(questionno_column),
        sg.Column(givans_column),
        sg.Column(actans_column)],

        [sg.Text('Score: ', size=(20, 1), font='Helvetica 14', key='score')]
    ]



    # create the window and show it without the plot
    window = sg.Window('Interview Platform',
                       layout, location=(0, 0))

    # ---===--- Event LOOP Read and display fr
    # ames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    isAnsweringQuestion = False
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    total_answers = []
    feedback_pointer = 0
    user_score = 0
    while True:
        event, values = window.read(timeout=1)
        ret, frame = cap.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray_img[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            # cv2.putText(frame, emotion_dict[maxindex], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        h, w, _ = frame.shape
        h = round(h / 2)
        w = round(w / 2)
        frame = cv2.resize(frame, (w, h))
        frame = cv2.flip(frame, 1)
        recordingOverlay(frame, '', (20, 50), (218, 18, 45))
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)



        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Start':
            window['Start'].update(disabled=True)
            window['Submit'].update(disabled=False)
            qu_lines = math.ceil(len(question) / 50)
            window['qu'].update(question, visible = True)
            window['qu'].set_size((50, qu_lines))
            start = time.time()
            if actual_difficulty == "Hard":
                window["i"].update(visible=True)
                window["ii"].update(visible=True)
            else:
                window["i"].update(visible=False)
                window["ii"].update(visible=False)
            window['C1'].update(text=options[0], visible=True, disabled=False)
            window['C2'].update(text=options[1], visible=True, disabled=False)
            window['C3'].update(text=options[2], visible=True, disabled=False)
            if len(options) == 6:
                window['C4'].update(text=options[3], visible=True, disabled=False)
                window['C5'].update(text=options[4], visible=True, disabled=False)
                window['C6'].update(text=options[5], visible=True, disabled=False)
            window.read(timeout=1)
            isAnsweringQuestion = True
            negative_frames = 0
            currentlyFeedbacking = False



        elif event == 'Next Question':
            print("Next question")
            window['C1'].update(visible=False, value=False)
            window['C2'].update(visible=False, value=False)
            window['C3'].update(visible=False, value=False)
            window['C4'].update(visible=False, value=False)
            window['C5'].update(visible=False, value=False)
            window['C6'].update(visible=False, value=False)
            window['qu'].update(visible=False)
            window["i"].update(visible=False)
            window["ii"].update(visible=False)

            question_number += 1
            if question_number == 7:

                # Append answers to emotion data
                answer_data = {
                    'question': [['1'],['2'],['3'],['4'],['5'],['6']],
                    'user answers': user_answers,
                    'actual answers': total_answers
                }
                print(len(question), len(user_answers), len(total_answers))

                res_df = pd.DataFrame(answer_data,
                                      columns=['question', 'user answers', 'actual answers'])

                bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
                file2 = os.path.abspath(os.path.join(bundle_dir, 'file_to_send_2.csv'))

                res_df.to_csv(file2)

                window['task'].update("See new window for your results.")
                window['Next Question'].update(visible=False)
                window['Start'].update(visible=False)
                window['Submit'].update(visible=True)
                window['question_number'].update(user_answers)
                window['difficulty'].update(visible=False)
                window['time'].update(visible=False)
                answer_window = sg.Window('Results', answer_layout, location=(0, 0))
                event, _ = answer_window.read(timeout=1)

                for i in range(1, 7):
                    if sorted(user_answers[i-1]) == sorted(total_answers[i-1]):
                        user_score +=1
                        answer_window['givans' + str(i)].update(text_color="green")
                    else:
                        answer_window['givans' + str(i)].update(text_color="red")

                    answer_to_print = '\n'.join(user_answers[i-1])
                    answer_window['givans' + str(i)].update(answer_to_print)
                    actual_answer_to_print = '\n'.join(total_answers[i-1])
                    answer_window['actans' + str(i)].update(actual_answer_to_print)

                answer_window['score'].update("Score: " + str(user_score) + " out of 6")


                while True:
                    event, values = answer_window.read(timeout=1)
                    if event == 'Exit' or event == sg.WIN_CLOSED:
                        return
            else:
                task, question, options, answers, actual_difficulty, label_difficulty, time_length = getNextQuestion(qu_df, question_number)
                options = options.split(', ')

                window['question_number']("Question " + str(question_number))

                # window['question_text'](question)
                window['task']("Task:\n" + task)
                window['difficulty'](label_difficulty)
                if label_difficulty == 'Hard':
                    difficulty_color = '#d12e30'
                else:
                    difficulty_color = '#33cc4f'
                window['difficulty'].update(background_color=difficulty_color)
                window['time']("Given time: " + time.strftime("%H:%M:%S", time.gmtime(time_length))[3:])
                window['Start'].update(disabled=False)
                window['Submit'].update(disabled=True)
                window['Next Question'].update(visible=False)


        if isAnsweringQuestion:
            # Facial expressions
            # total_answers, user_answers, feedback_pointer, window = answeringQuestion(window, time_length, cap, options, answers, question_number, user_answers, total_answers, feedback_pointer)
            # Audio
            changeKillAudioThread(False)
            t1 = Thread(target=audioAnalysis, daemon=True)
            t1.start()
            total_answers, user_answers, feedback_pointer = answeringQuestion(window, time_length, cap, options, answers, question_number, user_answers, total_answers, feedback_pointer)
            if total_answers == 0:
                return
            print("after facial, should now kill speech")
            changeKillAudioThread(True)
            isAnsweringQuestion = False
            print("alive? ", t1.isAlive())



def answeringQuestion(window, time_length, cap, options, answers, question_number, user_answers, total_answers, feedback_pointer):
    print("in answeringQuestion")
    start = time.time()
    currentlyFeedbacking = False
    isAnsweringQuestion = True
    negative_frames = 0
    audioFeedback = False
    while isAnsweringQuestion:
        event, values = window.read(timeout=2)  # run every 10 milliseconds

        current = time.time()
        seconds_left = time_length - (current - start)
        time_left = time.gmtime(seconds_left)
        time_left = time.strftime("%H:%M:%S", time_left)[3:]

        # updating video
        ret, frame = cap.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray_img[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            sum_positive = prediction[0][positive_indices].sum()
            sum_negative = prediction[0][negative_indices].sum()

            # If overall positive
            if sum_positive > sum_negative:
                # cv2.putText(frame, "positive " + emotion_dict[int(np.argmax(prediction))], (int(x), int(y)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (0, 0, 255),
                #             2)

                negative_frames = 0
                # print("positive ", prediction[0])
                if currentlyFeedbacking and not audioFeedback:
                    this_post_feedback.append(prediction[0].tolist())


            # If overall negative
            elif sum_positive < sum_negative:
                # cv2.putText(frame, "negative " + emotion_dict[int(np.argmax(prediction))], (int(x), int(y)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (0, 0, 255),
                #             2)
                negative_frames += 1
                # print("negative ", prediction[0])
                if currentlyFeedbacking and not audioFeedback:
                    this_post_feedback.append(prediction[0].tolist())
            print("audio_feedback_queue ", getAudioFeedbackQueue())

            # If 5+ negative consecutive frames, output feedback
            if (not currentlyFeedbacking and negative_frames >= 8) or getAudioFeedbackQueue() > 0:
                if getAudioFeedbackQueue() > 0:
                    audioFeedback = True
                    clearAudioFeedbackQueue()
                    print("audio_feedback_q post clear", getAudioFeedbackQueue())
                    pre_feedback.append(getNegativeResult(True))
                else:
                    audioFeedback = False
                    pre_feedback.append(prediction[0].tolist())
                this_post_feedback = []
                # cv2.putText(frame, "feedback ", (int(x), int(y)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (0, 0, 255),
                #             2)
                if feedback_pointer == 0:
                    window["feedback"].update("", visible=True)
                    feedback_type.append("No Feedback")
                else:
                    window["feedback"].update(random.choice(feedback_dictionary[feedback_pointer]))
                    if feedback_pointer == 1:
                        feedback_type.append("bald")
                    else:
                        feedback_type.append("positive_politeness")
                question_list.append(question_number)
                feedback_pointer = (feedback_pointer + 1) % 3

                currentlyFeedbacking = True
                finish_time = seconds_left - 5

            # Remove after 5 seconds or consistently happy
            if not audioFeedback and currentlyFeedbacking and seconds_left <= finish_time:
                window["feedback"].update("")
                negative_frames = 0
                currentlyFeedbacking = False
                post_feedback.append(this_post_feedback)
                this_post_feedback = []
                clearAudioFeedbackQueue()

            if audioFeedback and currentlyFeedbacking and len(getNegativeResult(False)) == 3:
                window["feedback"].update("")
                negative_frames = 0
                currentlyFeedbacking = False
                post_feedback.append(getNegativeResult(False))
                clearNegativeResult()
                this_post_feedback = []
                clearAudioFeedbackQueue()
                audioFeedback = False

        h, w, _ = frame.shape
        h = round(h / 2)
        w = round(w / 2)
        frame = cv2.resize(frame, (w, h))
        frame = cv2.flip(frame, 1)

        # draw the label into the frame
        if seconds_left > 10:
            recordingOverlay(frame, 'Time remaining: ' + time_left, (20, 50), (0, 0, 0))
        else:
            recordingOverlay(frame, 'Time remaining: ' + time_left, (20, 50), (0, 0, 255))

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)

        if event == "Submit" or seconds_left < 1:
            isAnsweringQuestion = False
            if currentlyFeedbacking:
                window["feedback"].update("")
                negative_frames = 0
                currentlyFeedbacking = False
                post_feedback.append(this_post_feedback)
                this_post_feedback = []

            window['C1'].update(disabled=True)
            window['C2'].update(disabled=True)
            window['C3'].update(disabled=True)
            window['C4'].update(disabled=True)
            window['C5'].update(disabled=True)
            window['C6'].update(disabled=True)
            window['Submit'].update(disabled=True)

            question_user_answers = []
            if values['C1']:
                question_user_answers.append(options[0])
            if values['C2']:
                question_user_answers.append(options[1])
            if values['C3']:
                question_user_answers.append(options[2])
            if values['C4']:
                question_user_answers.append(options[3])
            if values['C5']:
                question_user_answers.append(options[4])
            if values['C6']:
                question_user_answers.append(options[5])
            user_answers.append(question_user_answers)
            print("user_answers ", user_answers)

            answers = answers.split(', ')
            total_answers.append(answers)

            print('total_answers ', total_answers)

            emotion_data = {
                'question': question_list,
                'feedback type': feedback_type,
                'pre-feedback': pre_feedback,
                'post-feedback': post_feedback
            }

            # Takes into account if it didn't finish feedbacking
            if len(post_feedback) == len(pre_feedback) - 1:
                if audioFeedback:
                    post_feedback.append(getNegativeResult(False))
                else:
                    post_feedback.append(this_post_feedback)

            print(len(question_list), len(feedback_type), len(pre_feedback), len(post_feedback))
            res_df = pd.DataFrame(emotion_data, columns=['question', 'feedback type', 'pre-feedback', 'post-feedback'])

            bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
            file1 = os.path.abspath(os.path.join(bundle_dir, 'file_to_send_1.csv'))
            res_df.to_csv(file1)
            if question_number != 6:
                window['Next Question'].update(visible=True)
            else:
                window['Next Question'].update(visible=True)
                window['Next Question'].update("View Results")
            break

        if event == "Exit" or event == sg.WIN_CLOSED:
            return 0, 0, 0

    print("about to return")

    return total_answers, user_answers, feedback_pointer


def recordingOverlay(img, text, pos, col):
    font_face = cv2.FONT_HERSHEY_COMPLEX
    scale = 0.75
    color = col
    cv2.putText(img, text, pos, font_face, scale, color, 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font_face, scale, (255, 255, 255), 1, cv2.LINE_AA)

def readQuestionFile():
    import pandas

    bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    path_to_qu = os.path.abspath(os.path.join(bundle_dir, 'question-sheet.csv'))

    qu_df = pandas.read_csv(path_to_qu)
    return qu_df

def getNextQuestion(qu_df, num):
    # Question num is index + 1
    task = qu_df.at[num-1, 'Task']
    question = qu_df.at[num-1, 'Question']
    options = qu_df.at[num-1, 'Options']
    answers = qu_df.at[num - 1, 'Answers']
    label_difficulty = qu_df.at[num-1, 'Labelled difficulty']
    actual_difficulty = qu_df.at[num - 1, 'Actual difficulty']
    time_length = qu_df.at[num - 1, 'Timer length']
    return task, question, options, answers, actual_difficulty, label_difficulty, time_length




main()