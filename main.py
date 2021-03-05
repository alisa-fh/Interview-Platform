#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import numpy as np
import time

"""
Demo program that displays a webcam using OpenCV
"""


def main():

    sg.theme('Black')
    question_number = 1

    # define the window layout
    layout = [[sg.Text('Interview platform', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Text('Question ' + str(question_number), size=(40, 1), justification='left', font='Helvetica 15', key='question_number')],
              [sg.Image(filename='', key='image')],
              [sg.Button('Record', size=(10, 1), font='Helvetica 14'),
               sg.Button('Stop', size=(10, 1), font='Any 14', disabled=True),
               sg.Button('Next Question', size=(15, 1), font='Helvetica 14', visible=False),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14')
               ]]


    # create the window and show it without the plot
    window = sg.Window('Interview Platform',
                       layout, location=(300, 0))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    answeringQuestion = False

    while True:
        event, values = window.read(timeout=1)
        ret, frame = cap.read()
        h, w, _ = frame.shape
        h = h // 2
        w = w // 2
        frame = cv2.resize(frame, (w, h))
        recordingOverlay(frame, '', (20, 50), (218, 18, 45))
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)


        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Record':
            window['Record'].update(disabled=True)
            window['Stop'].update(disabled=False)
            start = time.time()
            answeringQuestion = True



        elif event == 'Next Question':
            question_number += 1
            window['question_number']("Question " + str(question_number))
            window['Record'].update(disabled=False)
            window['Stop'].update(disabled=True)
            window['Next Question'].update(visible=False)


        while answeringQuestion:
            event, values = window.Read(timeout=10)  # run every 10 milliseconds

            current = time.time()
            seconds_left = 20 - (current - start)
            time_left = time.gmtime(seconds_left)
            time_left = time.strftime("%H:%M:%S", time_left)[3:]

            # updating video
            ret, frame = cap.read()
            h, w, _ = frame.shape
            h = h // 2
            w = w // 2
            frame = cv2.resize(frame, (w, h))
            # draw the label into the frame
            if seconds_left > 10:
                recordingOverlay(frame, 'Time remaining: ' + time_left, (20, 50), (0, 0, 0))
            else:
                recordingOverlay(frame, 'Time remaining: ' + time_left, (20, 50), (0, 0, 255))

            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)



            if event == "Stop" or seconds_left < 1:
                answeringQuestion = False
                window['Stop'].update(disabled=True)
                window['Next Question'].update(visible=True)
                break

            if event == "Exit":
                return

def recordingOverlay(img, text, pos, col):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    color = col
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)




main()