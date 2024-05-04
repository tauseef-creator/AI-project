from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
import csv

app = Flask(__name__)

def load_labels():
    labels = {}
    with open('./training/labels.csv', mode='r') as file:
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
    model = load_model('./model/TSR.h5')
    data=[]
    image = Image.open(img).convert('RGB')
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    
    # Get the probabilities for each class
    probabilities = model.predict(X_test)
    
    # Convert probabilities to class labels
    Y_pred = np.argmax(probabilities, axis=1)   
    return Y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = "Predicted Traffic🚦Sign is: " +classes[a]
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
