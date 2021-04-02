from imutils.video import VideoStream
import imutils
import time
import cv2
import os
import PySimpleGUI as sg
from PIL import Image
import io


class faceCollector():
    def __init__(self, username):
        self.output_path = "custom-faces/" + username
        self.cascade = "newmodel/haarcascade_frontalface_default.xml"
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)

        # Window
        self.part_number = 1
        self.total_part = 8
        self.layout = [
                        [sg.Text('Part: ' + str(self.part_number), size=(10, 1), justification='center',
                                 font='Helvetica 20', key='part', background_color='#c1c1c1')],
                        [sg.Image(filename='', key='image')],
                        [sg.Text('', size=(50, 1), justification='center', font='Helvetica 14', key='caption')],
                        [sg.Button('Start', size=(10, 1), font='Helvetica 14'),
                        sg.Button('Next', size=(15, 1), font='Helvetica 14', visible=True, disabled=True),
                        sg.Button('Exit', size=(10, 1), font='Helvetica 14')]
                    ]
        sg.theme('DefaultNoMoreNagging')
        self.window = sg.Window('Preliminary Data Collection',
                           self.layout, location=(0, 0))
        self.emotiondata = ["angry", "neutral", "disgusted", "neutral2", "fear", "happy", "neutral3", "sad",]
        self.caption = ["Man at a concentration camp", "A stone wall", "Some dead bodies", "A brick wall", "A ghost", "Dog in a cup", "Cotton buds", "Destruction of a building"]


    def collect(self):
        # load OpenCV's Haar cascade for face detection from disk
        detector = cv2.CascadeClassifier(self.cascade)
        # initialize the video stream, allow the camera sensor to warm up,
        # and initialize the total number of example faces written to disk
        # thus far
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        # vs = VideoStream(usePiCamera=True).start()
        time.sleep(2.0)
        total = 0
        recording = False

        emotion_count = 0
        # loop over the frames from the video stream
        while True:
            event, values = self.window.read(timeout=1)
            if event == 'Exit' or event == sg.WIN_CLOSED:
                # print the total faces saved and do a bit of cleanup
                print("[INFO] {} face images stored".format(total))
                print("[INFO] cleaning up...")
                cv2.destroyAllWindows()
                vs.stop()
                return

            elif event == 'Start':
                self.window['Start'].update(disabled=True)
                self.window['Next'].update(disabled=False)
                self.window['Exit'].update(disabled=False)
                current_emotion = "./emotion-stimulus/" + self.emotiondata[emotion_count] + ".jpg"
                image = Image.open(current_emotion)
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                self.window["image"].update(data=bio.getvalue())
                self.window["image"].update(visible=True)
                self.window["caption"].update(self.caption[emotion_count])
                self.window["caption"].update(visible=True)
                current_dir = self.output_path + "/" + self.emotiondata[emotion_count]
                if not os.path.isdir(current_dir):
                    os.mkdir(current_dir)
                emotion_count += 1

                recording = True

            # grab the frame from the threaded video stream, clone it, (just
            # in case we want to write it to disk), and then resize the frame
            # so we can apply face detection faster
            frame_count = 0
            while recording == True:
                if frame_count < 101:
                    frame_count += 1
                event, values = self.window.read(timeout=10)
                frame = vs.read()
                orig = frame.copy()
                frame = imutils.resize(frame, width=400)
                # detect faces in the grayscale frame
                rects = detector.detectMultiScale(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
                    minNeighbors=5, minSize=(30, 30))
                # loop over the face detections and draw them on the frame
                for (x, y, w, h) in rects:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if len(rects) != 0:
                    roi = frame[y:y+h, x:x+w]
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    resized_roi_gray = cv2.resize(roi_gray, (40, 40), interpolation = cv2.INTER_AREA)
                    # show the output frame
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF

                    # if the `k` key was pressed, write the *original* frame to disk
                    # so we can later process it and use it for face recognition
                    # if key == ord("k"):
                    if frame_count % 5 == 0:
                        p = os.path.sep.join([current_dir, "{}.png".format(
                            str(total).zfill(5))])
                        cv2.imwrite(p, resized_roi_gray)
                        total += 1
                        print("Captured! (total:", total, ")")
                # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break
                if event == "Next":
                    print("event is next")
                    recording = False
                    self.window['Start'].update(disabled=False)
                    self.window['Next'].update(disabled=True)
                    self.window["image"].update(visible = False)
                    self.window["caption"].update(visible=False)
                    self.part_number += 1
                    print(self.part_number)
                    self.window['part']("Part " + str(self.part_number))
                    if self.part_number == self.total_part:
                        self.window['Exit']('Finish')
                        self.window['Exit'].update(disabled=True)
                        self.window["Next"].update(visible=False)

                if event == 'Exit' or event == sg.WIN_CLOSED:
                    # print the total faces saved and do a bit of cleanup
                    print("[INFO] {} face images stored".format(total))
                    print("[INFO] cleaning up...")
                    cv2.destroyAllWindows()
                    vs.stop()
                    return



alisas_face = faceCollector("alisa")
alisas_face.collect()
