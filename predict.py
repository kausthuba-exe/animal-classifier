""" Classifies a single image using the trained model.

Run:
    python predict.py path/to/image.jpg
"""
IMG_SIZE = (128,128)
import sys
import tensorflow as tf
IMG_SIZE = (128,128)
model = tf.keras.models.load_model("animal_classifier.keras")
class_names = open("class_names.txt").read().splitlines()

img_path = sys.argv[1]

img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
arr = tf.keras.utils.img_to_array(img)
arr = tf.expand_dims(arr, axis=0)
pred = model.predict(arr)[0]
best = pred.argmax()

print(f"Prediction: {class_names[best]}({pred[best] * 100:.1f}% confidence)")
for name, score in sorted(zip(class_names, pred), key=lambda x: -x[1]):
    print(f"{name}: {score * 100:.1f}%")
