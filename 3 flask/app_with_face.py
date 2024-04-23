from flask import Flask, render_template, Response, request
from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
import pickle
import time
import subprocess

import os

def run_flask_app():
    subprocess.Popen('python', "/home/kumar/Downloads/24 [BMSIT] DISEASE PREDICTION/3 flask/app.py")




app = Flask(__name__)

# Load face detector, face embedding model, recognizer, and label encoder
protoPath = "/home/kumar/Downloads/24 [BMSIT] DISEASE PREDICTION/2 face_detection_module/face_detector/deploy.prototxt"
modelPath = "/home/kumar/Downloads/24 [BMSIT] DISEASE PREDICTION/2 face_detection_module/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

embedder = cv2.dnn.readNetFromTorch("/home/kumar/Downloads/24 [BMSIT] DISEASE PREDICTION/2 face_detection_module/face_detector/openface_nn4.small2.v1.t7")

recognizer = pickle.loads(open("/home/kumar/Downloads/24 [BMSIT] DISEASE PREDICTION/2 face_detection_module/output/recognizer.pickle", "rb").read())
le = pickle.loads(open("/home/kumar/Downloads/24 [BMSIT] DISEASE PREDICTION/2 face_detection_module/output/le.pickle", "rb").read())

# Global variable to store the current class label
current_label = ""
entered_username = ""
username_printed = False

def generate_frames(entered_username):
    global current_label, username_printed
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    last_print_time = time.time()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                current_label = name  # Update the global variable with the current class label

                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                if entered_username != name:
                    cv2.putText(frame, "NOT MATCHED", (startX, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if entered_username == name:
                    cv2.putText(frame, "MATCHED", (startX, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check if the username has been entered and not printed yet
                if entered_username and not username_printed:
                    print("Username entered:", entered_username)
                    username_printed = True

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Check if 10 seconds have passed since the last print
        if time.time() - last_print_time >= 5:
            print("Current label:", current_label)
            last_print_time = time.time()

            print("entered username : ",entered_username)

            if current_label == entered_username:
                run_flask_app()
            #     print('MATCHED')
            # else:
            #     print('UNMATCHED')

    vs.stop()

@app.route('/')
def index():
    return render_template('index_7.html', current_label=current_label)

@app.route('/video_feed')
def video_feed():
    global entered_username
    return Response(generate_frames(entered_username), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/setUsername', methods=['POST'])
def set_username():
    global entered_username
    entered_username = request.json.get('username')
    return 'ok', 200

if __name__ == "__main__":
    app.run(debug=True, port=5005)
