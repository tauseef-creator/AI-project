import cv2
from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf

import csv

app = Flask(__name__)

def load_labels():
    labels = {}
    with open('./Training/labels.csv', mode='r') as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader)
        for row in reader:
            # Assuming the CSV file has two columns: index and label
            index = int(row[0])
            label = row[1]
            labels[index] = label
    return labels

# Call the function to load labels when the application starts
classes = load_labels()


def image_processing(img):
    model = load_model("./model/TSR.h5")
    data = []
    # image = Image.open(img).convert("L") # Convert image to grayscale
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))
    X_test = np.array(data)
    
    # Get the probabilities for each class
    probabilities = model.predict(X_test)
    print("Probabilities:", probabilities)

    # Convert probabilities to class labels
    Y_pred = np.argmax(probabilities, axis=1)
    print(Y_pred)

    # Determine the class with the highest probability
    max_prob_index = np.argmax(probabilities)
    max_prob = probabilities[0][max_prob_index]

    # Check if the model's confidence is below a threshold
    print(max_prob)
    confidence_threshold = 0.9 # You can adjust this value based on your needs
    if max_prob < confidence_threshold:
        return "Image not related to traffic signs"

    return "Predicted Traffic🚦Sign is: " + classes[Y_pred[0]]


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
        # Check if the result is a prediction or an error message
        print(result)
        if "Predicted Traffic🚦Sign is:" in result:
            pass
        else:
            result = "Image not related to traffic signs I am aware of!"
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
    # If the image is not in RGB format, convert it
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize the image to the expected dimensions
    img = cv2.resize(img, (30, 30))
    # img = img/255
    return img


@app.route("/video")
def video():
    return render_template("video.html")


model2 = load_model("./model/TSR.h5")

threshold = 0.9  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX


def generate_frames():
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)  # Set the resolution
    camera.set(4, 480)  # Set the resolution
    camera.set(10, 180)  # Brightness
    camera.set(15, -8.0)  # Exposure
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Disable auto-exposure
    camera.set(cv2.CAP_PROP_EXPOSURE, -3)  # Manually set exposure
    while True:
        success, imgOrignal = camera.read()

        # PROCESS IMAGE
        img = preprocessing(imgOrignal)
        # Create a window with the WINDOW_NORMAL flag
        img = img.reshape(1, 30, 30, 3)
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
        # Assuming 'img' is your preprocessed image with the correct shape
        predictions = model2.predict(img)

        # Get the class index with the highest probability
        classIndex = np.argmax(predictions, axis=1)

        # Get the probability value of the predicted class
        max_prod_index = np.argmax(predictions)
        max_prob = predictions[0][max_prod_index]

        if max_prob >= threshold:
            # Assuming getCalssName is a function that takes a class index and returns the class name
            class_name = classes[classIndex[0]]
            cv2.putText(
                imgOrignal,
                str(classIndex[0]) + " " + class_name,
                (120, 35),
                font,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                imgOrignal,
                str(round(max_prob * 100, 2)) + "%",
                (180, 75),
                font,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                imgOrignal,
                "No traffic sign detected",
                (120, 35),
                font,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        resized_img = cv2.resize(imgOrignal, (640, 480))
        # Encode the image as jpeg
        _, buffer = cv2.imencode(".jpg", resized_img)

        # Convert the buffer to bytes
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
