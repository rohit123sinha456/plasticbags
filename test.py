import tensorflow as tf
import cv2
import numpy as np
import os
if __name__=="__main__":
    os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    classes = ["cardboard","glass","metal","paper","plastic","trash"]
    print("Loading Model...")
    model = tf.keras.models.load_model('trained_model')
    print("Model is loaded")
    images_list = ['hwpb.jpeg','hwpb1.jpg','hwcb.jpeg','hwcb1.jpg','hcmc.jpg','hcpb.jpg']
    for image_name in images_list:
    # image_name = 'hwcb1.jpg'
        img = cv2.imread(image_name)
        imgp = cv2.resize(img,(512,512))
        cv2.imwrite('resized_image.jpg',imgp)
        imgp = np.expand_dims(imgp,axis=0)
        prediction = model.predict(imgp)
        print("the prediction for the image `{image_name}` is {pred}".format(image_name=image_name,pred=classes[np.argmax(prediction)]))
