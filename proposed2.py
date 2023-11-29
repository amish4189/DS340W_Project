import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import pandas as pd

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

def dft_processing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)

    radius = min(gray.shape) // 2
    cy, cx = np.indices(gray.shape)
    cy = cy - gray.shape[0] // 2
    cx = cx - gray.shape[1] // 2
    r = np.sqrt(cx**2 + cy**2).astype(int)

    radial_mean = np.bincount(r.ravel(), magnitude_spectrum.ravel()) / np.bincount(r.ravel())
    return radial_mean[:radius] / np.max(radial_mean)


def create_dft_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)  # Added dropout
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)  # Added batch normalization
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

def load_dataset(real_dir, fake_dir, frame_count=10, frame_size=(224, 224)):
    real_videos = [os.path.join(real_dir, fname) for fname in os.listdir(real_dir)]
    fake_videos = [os.path.join(fake_dir, fname) for fname in os.listdir(fake_dir)]

    X_frames = []
    y = []

    for video_path in real_videos + fake_videos:
        frame_chunks = preprocess_video(video_path, frame_count, frame_size)
        X_frames.extend(frame_chunks)
        y.extend([1 if video_path in real_videos else 0] * len(frame_chunks))

    return np.array(X_frames), np.array(y)


def load_dataset_with_dft(real_dir, fake_dir, frame_count=10, frame_size=(224, 224)):
    real_videos = [os.path.join(real_dir, fname) for fname in os.listdir(real_dir)]
    fake_videos = [os.path.join(fake_dir, fname) for fname in os.listdir(fake_dir)]

    X_dft = []
    y = []

    for video_path in real_videos + fake_videos:
        vidcap = cv2.VideoCapture(video_path)
        dft_features = []

        for i in range(frame_count):
            success, frame = vidcap.read()
            if not success:
                break

            frame_resized = cv2.resize(frame, frame_size)
            dft_feature = dft_processing(frame_resized)
            dft_features.append(dft_feature)

        vidcap.release()
        avg_dft_feature = np.mean(dft_features, axis=0) if dft_features else np.zeros((frame_size[0] // 2,))
        X_dft.append(avg_dft_feature)
        y.append(1 if video_path in real_videos else 0)

    return np.array(X_dft), np.array(y)

# Create a directory for extracted frames
frame_save_dir = "extracted_frames"
os.makedirs(frame_save_dir, exist_ok=True)

# Process and save frames from real videos
real_dir = 'data\\real'
fake_dir = 'data\\fake'
#for video_file in os.listdir(real_dir):
 #   video_path = os.path.join(real_dir, video_file)
  #  preprocess_video(video_path, save_frames=True, save_dir=frame_save_dir, label="real")

# Process and save frames from fake videos
#for video_file in os.listdir(fake_dir):
 #   video_path = os.path.join(fake_dir, video_file)
  #  preprocess_video(video_path, save_frames=True, save_dir=frame_save_dir, label="fake")

# Load the datasets
X_frames, y_frames = load_dataset(real_dir, fake_dir)
X_dft, y_dft = load_dataset_with_dft(real_dir, fake_dir)

# Split data into train, test, and unseen sets
X_frames_train, X_frames_temp, y_frames_train, y_frames_temp = train_test_split(X_frames, y_frames, test_size=0.3, random_state=42)
X_dft_train, X_dft_temp, y_dft_train, y_dft_temp = train_test_split(X_dft, y_dft, test_size=0.3, random_state=42)

# Ensure that both unseen datasets have the same length
min_length_unseen = min(len(X_frames_temp), len(X_dft_temp))
X_frames_unseen, y_frames_unseen = X_frames_temp[:min_length_unseen], y_frames_temp[:min_length_unseen]
X_dft_unseen, y_dft_unseen = X_dft_temp[:min_length_unseen], y_dft_temp[:min_length_unseen]

# Initialize and train both models
textural_model = create_textural_model((10, 224, 224, 3))
dft_model = create_dft_model((X_dft.shape[1],))

textural_model.fit(X_frames_train, y_frames_train, validation_data=(X_frames_temp, y_frames_temp), epochs=15, batch_size=16)
dft_model.fit(X_dft_train, y_dft_train, validation_data=(X_dft_temp, y_dft_temp), epochs=15, batch_size=32)

textural_accuracy_test = textural_model.evaluate(X_frames_temp, y_frames_temp)[1]
dft_accuracy_test = dft_model.evaluate(X_dft_temp, y_dft_temp)[1]

print(f"Textural Model Accuracy on Test Data: {textural_accuracy_test}")
print(f"DFT Model Accuracy on Test Data: {dft_accuracy_test}")

# Evaluate models on the unseen dataset
textural_accuracy_unseen = textural_model.evaluate(X_frames_unseen, y_frames_unseen)[1]
dft_accuracy_unseen = dft_model.evaluate(X_dft_unseen, y_dft_unseen)[1]

print(f"Textural Model Accuracy on Unseen Data: {textural_accuracy_unseen}")
print(f"DFT Model Accuracy on Unseen Data: {dft_accuracy_unseen}")

# Predictions on unseen data
textural_predictions_unseen = (textural_model.predict(X_frames_unseen) > 0.5).flatten()
dft_predictions_unseen = (dft_model.predict(X_dft_unseen) > 0.5).flatten()

# Convert boolean predictions to string labels for unseen data
textural_labels_unseen = np.where(textural_predictions_unseen, 'Real', 'Fake')
dft_labels_unseen = np.where(dft_predictions_unseen, 'Real', 'Fake')

# Decision Logic and Spreadsheet Creation for Unseen Data
results_unseen = pd.DataFrame({
    'Video_Frame_Prediction': textural_labels_unseen,
    'DFT_Prediction': dft_labels_unseen,
    'Final_Decision': np.where(textural_predictions_unseen == dft_predictions_unseen, textural_labels_unseen, 'Unsure')
})

results_unseen.to_csv('deepfake_detection_results_unseen.csv', index=False)
