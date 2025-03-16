from flask import Flask, render_template , request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


application = Flask(__name__)

model = VGG16()

@application.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

# upload images
@application.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile'] 
    image_path = "./images/"+ imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224,224))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    lable = decode_predictions(yhat)
    lable = lable[0][0]

    classification = '%s (%.2f%%)' % (lable[1], lable[2]*100)

    return render_template('index.html', prediction = classification)

if __name__ == '__main__':
    application.run(port=3000, debug=True)