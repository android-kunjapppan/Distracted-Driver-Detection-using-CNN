from keras.models import load_model
import tensorflow as tf
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


classes = {0: "safe driving",
1: "texting - right",
2: "talking on the phone - right",
3: "texting - left",
4: "talking on the phone - left",
5: "operating the radio",
6: "drinking",
7: "reaching behind",
8: "hair and makeup",
9: "talking to passenger",
}

#JSON
from keras.models import model_from_json

# Model reconstruction from JSON file
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('model_weights.h5')

# loading a keras model
# classifier=tf.keras.models.load_model("driverdistraction_lr_weights.h5.h5")
# ds_factor=0.6

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, frame = self.video.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
        new_img = cv2.resize(gray,(224,240))
        test = np.array(new_img.reshape(-1,240,240,3))
        preds = model.predict(test)
        preds=np.argmax(preds)
        for key,value in classes.items():
            if preds==key:
                predicted = value
        cv2.putText(frame,predicted,(50, 50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
