# CSCI 158 - Assignment 2
# Face Detection and Recognition using Convolutional Neural Networks (CNNs)
# This program performs face detection, crops faces, trains a CNN-based face recognition model,
# and evaluates its performance using a confusion matrix, accuracy, precision, recall, and F1-score.
# Author: Noah Wiley
# Date: 12/1/25

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

# Parameters
CURRDIR_PATH = os.path.dirname(os.path.abspath(__file__))
FACE_DETECTOR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_SIZE = (160, 160)
RAND_SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE
FINE_TUNE_AT = 100
MODELSAVE_PATH = os.path.join(CURRDIR_PATH, "face_recog_mobilenetv2.h5")
HISTORYSAVE_PATH = os.path.join(CURRDIR_PATH, "training_history.csv")

# Read CSV and Paths for Kaggle data
ARCHIVEDIR_PATH = os.path.join(CURRDIR_PATH, "archive")
CSV_PATH = os.path.join(ARCHIVEDIR_PATH, "Dataset.csv")
if not os.path.exists(ARCHIVEDIR_PATH):
    print(f"Archive directory not found. Creating: {ARCHIVEDIR_PATH}")
    os.makedirs(ARCHIVEDIR_PATH, exist_ok=True)

# Check if CSV file exists
if not os.path.exists(CSV_PATH):
    print(f"ERROR: Dataset.csv not found at {CSV_PATH}")
    print("\nPlease ensure:")
    print("1. You have downloaded the dataset")
    print("2. Extract it to the 'archive' folder in your project directory")
    print("3. The CSV file should be at: " + CSV_PATH)
    print("4. The images should be in: " + os.path.join(ARCHIVEDIR_PATH, "Faces", "Faces"))
    print("\nExpected structure:")
    print(f"  {CURRDIR_PATH}/")
    print(f"    archive/")
    print(f"      Dataset.csv")
    print(f"      Faces/")
    print(f"        Faces/")
    print(f"          [image files]")
    raise FileNotFoundError(f"Dataset.csv not found. Please download and extract the dataset to {ARCHIVEDIR_PATH}")

df = pd.read_csv(CSV_PATH)
print("CSV loaded, rows: ", len(df))
print(df.head())

df.rename(columns={"id": "image_path"}, inplace=True)

# Face Images Path and paths to each image
FACES_DIR = os.path.join(ARCHIVEDIR_PATH, "Faces", "Faces")
if not os.path.exists(FACES_DIR):
    print(f"ERROR: Faces directory not found at {FACES_DIR}")
    print("\nPlease ensure the images are extracted to the correct location.")
    raise FileNotFoundError(f"Faces directory not found at {FACES_DIR}")

df['image_path'] = df['image_path'].apply(lambda fname: os.path.join(FACES_DIR, fname))

print("\nExample image paths:")
print(df['image_path'].head())
missing = df[~df['image_path'].apply(os.path.exists)]
if len(missing) > 0:
    print("\nFiles missing: ", len(missing))
    print(missing.head())
    print("\nWARNING: Some image files are missing. The model will skip these images.")
else:
    print("\nNo missing files found.")

if 'image_path' not in df.columns or 'label' not in df.columns:
    raise ValueError("CSV must have 'image_path' and 'label' columns.")

# Face Detection
face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)


def detect_and_crop_face(img_path, target_size=IMG_SIZE, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return None

    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    if len(faces) == 0:
        h, w = img.shape[:2]
        s = min(h, w)
        cy, cx = h // 2, w // 2
        y1, x1 = max(0, cy - s // 2), max(0, cx - s // 2)
        crop = img[y1:y1 + s, x1:x1 + s]
    else:
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        x, y, w, h = faces[0]
        pad = int(0.2 * max(w, h))
        y1 = max(0, y - pad)
        x1 = max(0, x - pad)
        y2 = min(img.shape[0], y + h + pad)
        x2 = min(img.shape[1], x + w + pad)
        crop = img[y1:y2, x1:x2]

    try:
        crop = cv2.resize(crop, target_size)
    except:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop


images = []
labels = []
failed = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    p = row['image_path']
    lbl = row['label']
    img = detect_and_crop_face(p, target_size=IMG_SIZE)
    if img is None:
        failed += 1
        continue
    images.append(img)
    labels.append(lbl)

print(f"Number of images procesed = {len(images)}, failed = {failed}")

if len(images) == 0:
    raise RuntimeError("No images were successfully processed.")

X = np.array(images, dtype=np.float32)

X /= 255.0

le = LabelEncoder()
y = le.fit_transform(labels)
num_classes = len(le.classes_)
print("Number of classes = ", num_classes)
print("Classes: ", le.classes_)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RAND_SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RAND_SEED)

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.08, height_shift_range=0.08, zoom_range=0.1,
                                   horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow(X_train, tf.keras.utils.to_categorical(y_train, num_classes),
                                     batch_size=BATCH_SIZE, shuffle=True)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, tf.keras.utils.to_categorical(y_val, num_classes))).batch(
    BATCH_SIZE).prefetch(AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, tf.keras.utils.to_categorical(y_test, num_classes))).batch(
    BATCH_SIZE).prefetch(AUTOTUNE)

base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = inputs
x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(MODELSAVE_PATH, monitor='val_loss', save_best_only=True)
reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_dataset,
                    callbacks=[earlystopping, checkpoint, reduceLR])

base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

fine_tune_epochs = 10
history_ft = model.fit(train_generator, epochs=fine_tune_epochs, validation_data=val_dataset,
                       callbacks=[earlystopping, checkpoint, reduceLR])

hist_df = pd.DataFrame(history.history)
if history_ft:
    hist_df = pd.concat([hist_df, pd.DataFrame(history_ft.history)], ignore_index=True)
hist_df.to_csv(HISTORYSAVE_PATH, index=False)

y_pred_proba = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = y_test

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f"Test Accuracy: {acc:.4f} Precision: {prec:.4f} Recall: {rec:.4f} F1 (weighted): {f1:.4f}")

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix shape = ", cm.shape)

report = classification_report(y_true, y_pred, target_names=le.classes_)
print(report)

model.save(MODELSAVE_PATH)
print("Model can be found in: ", MODELSAVE_PATH)
