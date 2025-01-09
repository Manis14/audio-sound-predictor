import streamlit as st
from prediction import predict_audio_class

# Streamlit app layout
st.title("Audio Classification with Deep Learning")

# Information about the model's capabilities
st.write("This model classifies audio into the following categories:")
classes = ['Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark',
           'Drilling', 'Engine Idling', 'Gun Shot', 'Jackhammer', 'Siren',
           'Street Music']

# Display the classes in a user-friendly way
st.write("**Class Labels:**")
st.write(", ".join(classes))

st.write("\nUpload a .wav audio file, and the model will predict the class of the sound.")

# Upload audio file
audio_file = st.file_uploader("Upload an audio file (.wav only)", type=['wav'])

# When a file is uploaded
if audio_file is not None:
    # Save the uploaded file
    file_path = "uploaded_audio.wav"
    with open(file_path, "wb") as f:
        f.write(audio_file.getbuffer())

    # Play the uploaded audio file
    st.audio(file_path, format="audio/wav")

    # Get the prediction
    prediction = predict_audio_class(file_path)

    if prediction == "Please provide a .wav file only.":
        st.error(prediction)
    else:
        st.success(f"The predicted sound class is: **{prediction}**")
