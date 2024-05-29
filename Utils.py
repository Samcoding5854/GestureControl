import tensorflow as tf
import numpy as np
import cv2
from keras.models import model_from_json

## Emotion Detector ##

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0


json_file = open("emotiondetectornew1.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetectornew1.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)


labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
def EmotionDetector(im):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    for (p,q,r,s) in faces:
        image = gray[q:q+s,p:p+r]
        cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
        image = cv2.resize(image,(48,48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        # print("Predicted Output:", prediction_label)
        # cv2.putText(im,prediction_label)
        # cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
        return prediction_label
 

