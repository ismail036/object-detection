from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
translate = ["DOG", "HORSE", "ELEPHANT", "BUTTERFLY",'FEMALE', "CHICKEN", "CAT",'MALE', "COW", "SHEEP", "SPIDER",'RANDOM' , "SQUIRREL"]


from flask import Flask, jsonify, send_from_directory, url_for

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Dosya adı geçersiz'}), 400

    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img6 = image.load_img(rf'uploads/{filename}', target_size= (224, 224))
        img6 = image.img_to_array(img6)
        img6 = np.expand_dims(img6,axis=0)
        model = load_model('animal-10.hdf5')
        result_sp = model.predict(img6)
        print(translate[np.argmax(result_sp[0])])
        return jsonify({'success': 'Dosya yüklendi', 'filename': filename , 'type' :translate[np.argmax(result_sp[0])] }), 200

@app.route('/index.html')
def index2():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/members.html')
def members():
    return render_template('members.html')

@app.route('/images')
def get_images():
    image_list = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            download_url = url_for('get_image', filename=filename)
            image_list.append({'name': filename, 'download_url': download_url})
    return jsonify({'images': image_list})

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run()
