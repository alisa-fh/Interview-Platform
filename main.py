#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import numpy as np
import time

"""
Demo program that displays a webcam using OpenCV
"""


def main():
    df = readQuestionFile()
    print(df)

    sg.theme('DefaultNoMoreNagging')
    question_number = 1
    task, question, options, answers, label_difficulty, time_length = getNextQuestion(df, question_number)
    options = options.split(', ')

    if label_difficulty == 'Hard':
        difficulty_color = '#d12e30'
    else:
        difficulty_color = '#33cc4f'

    left_column = [[sg.Text('Task:\n ' + task, size=(50, 4), justification='center', font='Helvetica 20', key='task',
                            background_color='#c1c1c1')],

                   [sg.Text()],

                   [sg.Text('Question ' + str(question_number), size=(8, 1), font='Helvetica 20', key='question_number'),
                    sg.Text(label_difficulty, size=(4, 1), font='Helvetica 20', justification= "left", key='difficulty',
                            background_color= difficulty_color),
                    sg.Text('Given time: ' + time.strftime("%H:%M:%S", time.gmtime(time_length))[3:], size=(15, 1),
                            font='Helvetica 20', justification="left", key='time')
                    ],

                    [sg.Text(question, size=(50, 8), justification='center', font='Helvetica 20',
                        key='question_text')],

                   [sg.Checkbox('Option 1', font='Helvetica 14', key='C1', visible = False),
                   sg.Checkbox('Option 2', font='Helvetica 14', key='C2', visible = False),
                   sg.Checkbox('Option 3', font='Helvetica 14', key='C3', visible = False)],

                   [sg.Checkbox('Option 4', font='Helvetica 14', key='C4', visible = False),
                    sg.Checkbox('Option 5', font='Helvetica 14', key='C5', visible= False),
                    sg.Checkbox('Option 6', font='Helvetica 14', key='C6', visible=False)]

                   ]

    right_column = [[sg.Image(filename='', key='image')]]

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

    while True:
        event, values = window.read(timeout=1)
        ret, frame = cap.read()
        h, w, _ = frame.shape
        h = round(h / 2)
        w = round(w / 2)
        frame = cv2.resize(frame, (w, h))
        frame = cv2.flip(frame, 1)
        recordingOverlay(frame, '', (20, 50), (218, 18, 45))
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)
        window['question_text'].update(visible=False)


        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Start':
            window['Start'].update(disabled=True)
            window['Submit'].update(disabled=False)
            window['question_text'].update(visible = True)
            start = time.time()
            answeringQuestion = True



        elif event == 'Next Question':
            question_number += 1
            task, question, options, answers, label_difficulty, time_length = getNextQuestion(df, question_number)
            options = options.split(', ')
            window['question_number']("Question " + str(question_number))
            window['question_text'](question)
            window['task']("Task:\n" + task)
            window['difficulty'](label_difficulty)
            if label_difficulty == 'Hard':
                difficulty_color = '#d12e30'
            else:
                difficulty_color = '#33cc4f'
            window['difficulty'].update(background_color=difficulty_color)
            window['time']("Given time: " + time.strftime("%H:%M:%S", time.gmtime(time_length))[3:])
            window['C1'].update(text = options[0], visible = "True")
            window['C2'].update(text=options[1], visible = "True" )
            window['C3'].update(text=options[2], visible = "True")
            if len(options) == 6:
                window['C4'].update(text=options[3], visible = "True")
                window['C5'].update(text=options[4], visible = "True")
                window['C6'].update(text=options[5], visible = "True")


            window['Start'].update(disabled=False)
            window['Submit'].update(disabled=True)
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
    time_length = df.at[num - 1, 'Timer length']
    return task, question, options, answers, label_difficulty, time_length








main()