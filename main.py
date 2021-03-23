#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import numpy as np
import time
import math

import os
from keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from keras.preprocessing import image
import imutils

"""
Demo program that displays a webcam using OpenCV
"""

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


#load weights
model.load_weights('./newmodel/model.h5')
# load facial detection
face_haar_cascade = cv2.CascadeClassifier('./newmodel/haarcascade_frontalface_default.xml')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
negative_indices = [0, 1, 2]
positive_indices = [3, 4, 6]

def main():
    df = readQuestionFile()
    print(df)

    sg.theme('DefaultNoMoreNagging')
    question_number = 1
    task, question, options, answers, actual_difficulty, label_difficulty, time_length = getNextQuestion(df, question_number)
    options = options.split(', ')

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

    left_column = [[sg.Text('Task:\n ' + task, size=(50, 4), justification='center', font='Helvetica 20', key='task',
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
                    [sg.Text("\n\n\n", size=(50, 4),
                             font='Helvetica 18', justification="left", key='feedback', visible=True)]
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
        sg.Button('Exit', size=(10, 1), font='Helvetica 14')]
        ]



    # create the window and show it without the plot
    window = sg.Window('Interview Platform',
                       layout, location=(0, 0))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    answeringQuestion = False
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


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
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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

            answeringQuestion = True
            negative_frames = 0
            currentlyFeedbacking = False



        elif event == 'Next Question':
            question_number += 1
            task, question, options, answers, actual_difficulty, label_difficulty, time_length = getNextQuestion(df, question_number)
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
            window['C1'].update(visible = False, value=False)
            window['C2'].update(visible = False, value=False)
            window['C3'].update(visible = False, value=False)
            window['C4'].update(visible = False, value=False)
            window['C5'].update( visible = False, value=False)
            window['C6'].update( visible = False, value=False)
            window['qu'].update(visible=False)
            window["i"].update(visible=False)
            window["ii"].update(visible=False)



            window['Start'].update(disabled=False)
            window['Submit'].update(disabled=True)
            window['Next Question'].update(visible=False)

            if question_number == 6:
                window['Exit'].update(disabled=True)
                window['Exit']('Finish')


        while answeringQuestion:
            event, values = window.Read(timeout=10)  # run every 10 milliseconds

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
                    cv2.putText(frame, "positive "+ emotion_dict[int(np.argmax(prediction))], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255),
                                2)

                    negative_frames = 0

                # If overall negative
                else:
                    cv2.putText(frame, "negative " + emotion_dict[int(np.argmax(prediction))], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255),
                                2)
                    negative_frames += 1

                # If 5+ negative consecutive frames, output feedback
                if not currentlyFeedbacking and negative_frames >= 5:
                    cv2.putText(frame, "feedback ", (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255),
                                2)
                    window["feedback"].update("Here is some feedback", visible=True)
                    currentlyFeedbacking = True
                    finish_time = seconds_left - 5

                # Remove after 5 seconds or consistently happy

                if currentlyFeedbacking and seconds_left <= finish_time:
                    window["feedback"].update("\n\n\n")
                    negative_frames = 0
                    currentlyFeedbacking = False



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
                answeringQuestion = False
                window['C1'].update(disabled=True)
                window['C2'].update(disabled=True)
                window['C3'].update(disabled=True)
                window['C4'].update(disabled=True)
                window['C5'].update(disabled=True)
                window['C6'].update(disabled=True)
                window['Submit'].update(disabled=True)
                if question_number != 6:
                    window['Next Question'].update(visible=True)
                else:
                    window['Exit'].update(disabled=False)
                break

            if event == "Exit":
                return

def recordingOverlay(img, text, pos, col):
    font_face = cv2.FONT_HERSHEY_COMPLEX
    scale = 0.75
    color = col
    cv2.putText(img, text, pos, font_face, scale, color, 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font_face, scale, (255, 255, 255), 1, cv2.LINE_AA)

def readQuestionFile():
    import pandas
    df = pandas.read_csv('question-sheet.csv')
    return df

def getNextQuestion(df, num):
    # Question num is index + 1
    task = df.at[num-1, 'Task']
    question = df.at[num-1, 'Question']
    options = df.at[num-1, 'Options']
    answers = df.at[num - 1, 'Answers']
    label_difficulty = df.at[num-1, 'Labelled difficulty']
    actual_difficulty = df.at[num - 1, 'Actual difficulty']
    time_length = df.at[num - 1, 'Timer length']
    return task, question, options, answers, actual_difficulty, label_difficulty, time_length



main()