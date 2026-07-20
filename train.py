""" Trains a small CNN to classify animals: dog, cat, human, pig, lion, tiger.
Expected folder layout:
    data/
        dog/    *.jpg
        cat/    *.jpg
        human/  *.jpg
        pig/    *.jpg
        lion/   *.jpg
        tiger/  *.jpg
Run:
    python train.py
"""
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers, models
DATA_DIR = "data"
IMG_SIZE = (128,128)
BATCH_SIZE = 32
EPOCHS = 15

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split = 0.2, seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, subset= "training"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split = 0.2, seed=42,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, subset = "validation"
)
class_names = train_ds.class_names
print ("Classes:", class_names)
#cache/prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

#Small CNN
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(*IMG_SIZE, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Conv2D(32, 3 , activation='relu'),
    layers.Conv2D(64, 3 , activation='relu'),
    layers.Conv2D(128, 3 , activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(train_ds,validation_data=val_ds,epochs=EPOCHS)
model.save("animal_classidier.keras")
with open("class_names.txt","w")as f:
    f.write("\n".join(class_names))
print("Saved model to /animal_classidier.keras")