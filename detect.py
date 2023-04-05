import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

path = "E:/mypython/riceProject/model/7"
model = load_model(path)

class_names = ["BLB", "BPH", "Brown_Spot", "False_Smut", "Healthy_plant", "Hispa", "Neck_Blast", "Sheath_Blight_Rot", "Stemborer"]

img_root = "C:/Users/User/Documents/rice diseases/Resized_Original_Data/"
img_path = "Stemborer/Stemborer1_1ce1fae7-0bc4-4fb8-aee1-f164fc491976.jpeg"

img_path_file = img_root + img_path

rice_leaf = cv2.imread(img_path_file)
b, g, r = cv2.split(rice_leaf)       # get b,g,r
rice_leaf = cv2.merge([r, g, b])     # switch it to rgb

test_image = cv2.resize(rice_leaf, (224, 224))
test_image = img_to_array(test_image)  # convert image to np array
target_image = test_image.copy()
test_image = tf.expand_dims(test_image, 0)  # change dimension 3D to 4D

result = model.predict(test_image)  # predict diseased plant or not
# print("the result :", result)

pre = np.argmax(result[0])  # two d array
# print(pre)
print("confidence :", np.max(result[0])*100, " %")
print("predicted class :", class_names[pre])

actual_class = img_path.split('/')
print("actual class :", actual_class[0])
