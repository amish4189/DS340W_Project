import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import pandas as pd


def extract_facial_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces) > 0:
        face = faces[0]  # Considering the first detected face
        landmarks = shape_predictor(gray, face)
        facial_features = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)]
        return facial_features
    else:
        return None

def preprocess_video(video_path, frame_count=10, frame_size=(224, 224), save_frames=False, save_dir="extracted_frames", label=""):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    video_name = os.path.basename(video_path).split('.')[0]  # Extract video name
    label_dir = os.path.join(save_dir, label)  # Subdirectory for real or fake

    if save_frames and not os.path.exists(label_dir):
        os.makedirs(label_dir, exist_ok=True)

    for i in range(total_frames):
        success, frame = vidcap.read()
        if success:
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
            if save_frames:
                frame_filename = f"{video_name}_frame_{i}.jpg"
                cv2.imwrite(os.path.join(label_dir, frame_filename), frame)
    vidcap.release()

    return [frames[i:i+frame_count] for i in range(0, len(frames), frame_count) if i+frame_count <= len(frames)]

def load_dataset(real_dir, fake_dir):
    real_videos = [os.path.join(real_dir, fname) for fname in os.listdir(real_dir)]
    fake_videos = [os.path.join(fake_dir, fname) for fname in os.listdir(fake_dir)]

    X_frames = []
    y = []

    for vid in real_videos + fake_videos:
        frame_chunks = preprocess_video(vid)
        X_frames.extend(frame_chunks)
        y.extend([1 if vid in real_videos else 0] * len(frame_chunks))

    return np.array(X_frames), np.array(y)

def create_textural_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the dataset
real_dir = 'data\\real'
fake_dir = 'data\\fake'

# Create a directory for extracted frames
frame_save_dir = "extracted_frames"
os.makedirs(frame_save_dir, exist_ok=True)

# Process and save frames from real videos
#for video_file in os.listdir(real_dir):
 #   video_path = os.path.join(real_dir, video_file)
  #  preprocess_video(video_path, save_frames=True, save_dir=frame_save_dir, label="real")

# Process and save frames from fake videos
#for video_file in os.listdir(fake_dir):
 #   video_path = os.path.join(fake_dir, video_file)
  #  preprocess_video(video_path, save_frames=True, save_dir=frame_save_dir, label="fake")

# Load the dataset
X_frames, y = load_dataset(real_dir, fake_dir)

# Split data into train, test, and unseen sets
X_frames_train, X_frames_temp, y_frames_train, y_frames_temp = train_test_split(X_frames, y, test_size=0.4, random_state=42)
X_frames_test, X_frames_unseen, y_frames_test, y_frames_unseen = train_test_split(X_frames_temp, y_frames_temp, test_size=0.5, random_state=42)

# Initialize and train the textural model
textural_model = create_textural_model((10, 224, 224, 3))
textural_model.fit(X_frames_train, y_frames_train, validation_data=(X_frames_test, y_frames_test), epochs=10, batch_size=16)

# Evaluate the model on the test dataset
textural_accuracy_test = textural_model.evaluate(X_frames_test, y_frames_test)[1]
print(f"Textural Model Accuracy on Test Data: {textural_accuracy_test}")

# Evaluate the model on the unseen dataset
textural_accuracy_unseen = textural_model.evaluate(X_frames_unseen, y_frames_unseen)[1]
print(f"Textural Model Accuracy on Unseen Data: {textural_accuracy_unseen}")

# Predictions on unseen data
textural_predictions_unseen = (textural_model.predict(X_frames_unseen) > 0.5).flatten()

# Convert boolean predictions to string labels for unseen data
textural_labels_unseen = np.where(textural_predictions_unseen, 'Real', 'Fake')

# Decision Logic and Spreadsheet Creation for Unseen Data
results_unseen = pd.DataFrame({
    'Video_Frame_Prediction': textural_labels_unseen,
    'Final_Decision': textural_labels_unseen
})

results_unseen.to_csv('deepfake_detection_results_unseen.csv', index=False)
