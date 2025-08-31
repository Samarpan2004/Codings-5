# emotion_cnn.py
# This is a small CNN. Train on FER-like data (grayscale 48x48 images) or use transfer learning for better results.
# For demonstration create train/val folders with subfolders ['angry','happy',...]
# Run: python emotion_cnn.py train_dir val_dir

import sys, os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

def build_model(num_classes):
    model = models.Sequential([
        layers.Input((48,48,1)),
        layers.Conv2D(32,(3,3),activation='relu'), layers.MaxPool2D(),
        layers.Conv2D(64,(3,3),activation='relu'), layers.MaxPool2D(),
        layers.Conv2D(128,(3,3),activation='relu'), layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    if len(sys.argv)<3:
        print("Usage: emotion_cnn.py train_dir val_dir"); return
    train_dir, val_dir = sys.argv[1], sys.argv[2]
    datagen = ImageDataGenerator(rescale=1./255)
    train = datagen.flow_from_directory(train_dir, target_size=(48,48), color_mode='grayscale', batch_size=32, class_mode='categorical')
    val = datagen.flow_from_directory(val_dir, target_size=(48,48), color_mode='grayscale', batch_size=32, class_mode='categorical')
    model = build_model(train.num_classes)
    model.fit(train, validation_data=val, epochs=20)
    model.save("models/emotion_cnn.h5")
    print("Saved emotion model.")

if __name__=="__main__":
    main()
