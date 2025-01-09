import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('Audio_classifier.h5')

# Class names
classes = ['Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark',
           'Drilling', 'Engine Idling', 'Gun Shot', 'Jackhammer', 'Siren',
           'Street Music']


def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features


def predict_audio_class(file_name):
    # Check file extension
    if not file_name.lower().endswith('.wav'):
        return "Please provide a .wav file only."

    # Extract features
    features = features_extractor(file_name)

    # Reshape the feature to match the model's expected input shape
    features = features.reshape(1, -1)

    # Predict class probabilities
    predictions = model.predict(features)

    # Get the index of the highest probability (predicted class)
    predicted_class = np.argmax(predictions, axis=1)

    # Return the predicted class label
    return classes[predicted_class[0]]
