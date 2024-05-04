import cv2
from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf


app = Flask(__name__)

# Classes of trafic signs
classes = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicle > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing vehicle > 3.5 tons",
}


def image_processing(img):
    model = load_model("AI-project/model/TSR.h5")
    data = []
    image = Image.open(img).convert("RGB")
    image = image.resize((30, 30))
    data.append(np.array(image))
    X_test = np.array(data)

    # Get the probabilities for each class
    probabilities = model.predict(X_test)

    # Convert probabilities to class labels
    Y_pred = np.argmax(probabilities, axis=1)
    return Y_pred


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Get the file from post request
        f = request.files["file"]
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = "Predicted Traffic🚦Sign is: " + classes[a]
        os.remove(file_path)
        return result
    return None


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


@app.route("/video")
def video():
    return render_template("video.html")


model2 = load_model("AI-project/model/TSR.h5")

threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX


def generate_frames():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    camera.set(10, 180)
    camera.set(15, -8.0)
    while True:
        success, imgOrignal = camera.read()
        if not success:
            break
        else:
            # ret, buffer = cv2.imencode('.jpg', frame)
            # frame = buffer.tobytes()
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # PROCESS IMAGE
            img = np.asarray(imgOrignal)
            img = cv2.resize(img, (32, 32))
            img = preprocessing(img)
            cv2.imshow("Processed Image", img)
            img = img.reshape(1, 32, 32, 1)
            cv2.putText(
                imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                imgOrignal,
                "PROBABILITY: ",
                (20, 75),
                font,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            # PREDICT IMAGE
            predictions = model2.predict(img)
            classIndex = model2.predict_classes(img)
            probabilityValue = np.amax(predictions)
            if probabilityValue > threshold:
                # print(getCalssName(classIndex))
                # cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    imgOrignal,
                    str(classIndex) + " " + str(classes[classIndex[0]]),
                    (120, 35),
                    font,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    imgOrignal,
                    str(round(probabilityValue * 100, 2)) + "%",
                    (180, 75),
                    font,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Result", imgOrignal)

            if cv2.waitKey(1) and 0xFF == ord("q"):
                break


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(debug=True)