# UrbanSoundAI: Audio Classification App

UrbanSoundAI is a deep learning-based web application for classifying urban sounds. This application uses a trained model to predict sound classes from uploaded `.wav` audio files. The app is built using **Streamlit** for the frontend and supports interactive audio playback and prediction.

---

## Features

- **Supported Classes**:
  The model can classify audio into the following categories:
  - Air Conditioner
  - Car Horn
  - Children Playing
  - Dog Bark
  - Drilling
  - Engine Idling
  - Gun Shot
  - Jackhammer
  - Siren
  - Street Music

- **Interactive Web Interface**:
  - Upload `.wav` audio files.
  - Play the uploaded audio file directly in the app.
  - Get predictions for the sound class.

---

## Model Performance

- **Artificial Neural Network (ANN)**:
  - Achieved an accuracy of **73.84% (0.7384)**.
  - This architecture proved to be more effective for this dataset.

- **Recurrent Neural Network (RNN)**:
  - Achieved an accuracy of **61.05% (0.6105)**.
  - Despite being designed for sequential data, the RNN model performed worse than ANN on this task.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/urban-sound-classifier.git 
  
2. Navigate to the project directory:
   ```bash
   cd urban-sound-classifier
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the Streamlit app:
   ```bash
   streamlit run app.py

## Acknowledgments
- UrbanSound8K Dataset: Used for training and testing the model.
- Streamlit: For building the interactive web interface.
- Librosa: For audio feature extraction.





