import pandas as pd
import numpy as np

import librosa
from scipy.stats import skew, kurtosis, entropy, mode
import aubio
import joblib

import tkinter as tk
from tkinter import filedialog

# Load the audio file

def extract_features(audio_file_path):

    audio_data, sample_rate = librosa.load(audio_file_path)

    # ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',
    # 'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',
    # 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx', 'label']

    # Extract features using Librosa
    meanfreq = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate).mean()
    sd = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate).std()
    median = np.median(audio_data)
    Q25 = np.percentile(audio_data, 25)
    Q75 = np.percentile(audio_data, 75)
    IQR = Q75 - Q25
    skewness = skew(audio_data)
    kurt = kurtosis(audio_data)
    sp_ent = entropy(np.abs(audio_data))
    sfm = librosa.feature.spectral_flatness(y=audio_data).mean()
    mode_value, _ = mode(audio_data, keepdims=True)
    mode_ = float(mode_value[0])
    centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate).mean()
    meanfun = librosa.effects.harmonic(audio_data).mean()
    minfun = librosa.effects.preemphasis(audio_data).min()
    maxfun = librosa.effects.preemphasis(audio_data).max()
    meandom = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate).mean()
    mindom = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate).min()
    maxdom = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate).max()
    dfrange = maxdom - mindom


    # Use aubio for pitch estimation
    pitch_o = aubio.pitch("yin", 2048, 2048, sample_rate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(0.8)

    # Initialize an array to store pitch values
    pitches = []

    # Process audio data and extract pitch values
    hop_size = 2048
    total_frames = len(audio_data) // hop_size
    for frame in range(total_frames):
        samples = audio_data[frame * hop_size: (frame + 1) * hop_size]
        pitch = pitch_o(samples)[0]
        confidence = pitch_o.get_confidence()
        
        # Append pitch values to the array
        if confidence > 0.5:  # Adjust the confidence threshold as needed
            pitches.append(pitch)
        else:
            pitches.append(np.nan)

    # Use the median pitch value as modindx
    modindx = np.nanmedian(pitches)

    # Create a DataFrame with the extracted features
    new_voice_sample = pd.DataFrame({
        'meanfreq': meanfreq,
        'sd': sd,
        'median': median,
        'Q25': Q25,
        'Q75': Q75,
        'IQR': IQR,
        'skew': skewness,
        'kurt': kurt,
        'sp.ent': sp_ent,
        'sfm': sfm,
        'mode': mode_,
        'centroid': centroid,
        'meanfun': meanfun,
        'minfun': minfun,
        'maxfun': maxfun,
        'meandom': meandom,
        'mindom': mindom,
        'maxdom': maxdom,
        'dfrange': dfrange,
        'modindx': modindx
    }, index=[0])  # Use index=[0] to create a single-row DataFrame

    return new_voice_sample

def browse_audio_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
    if file_path:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, file_path)

def predict_label():
    audio_file_path = entry_path.get()
    if audio_file_path:
        new_voice_sample = extract_features(audio_file_path)
        prediction = svm_model.predict(new_voice_sample)
        if prediction[0] == 0:
            label_result.config(text=f"Predicted Label: Female")
        else:
            label_result.config(text=f"Predicted Label: Male")
    else:
        label_result.config(text="Please select an audio file.")

# Load the pre-trained KNN model
svm_model = joblib.load('svm_model.pkl')

# Create the main window
window = tk.Tk()
window.title("Voice Label Prediction")

# Create GUI components
label_instruction = tk.Label(window, text="Select an audio file:")
entry_path = tk.Entry(window, width=40)
button_browse = tk.Button(window, text="Browse", command=browse_audio_file)
button_predict = tk.Button(window, text="Predict Gender", command=predict_label)
label_result = tk.Label(window, text="Predicted Gender: ")

# Arrange GUI components using the grid layout
label_instruction.grid(row=0, column=0, columnspan=2, pady=10)
entry_path.grid(row=1, column=0, padx=10)
button_browse.grid(row=1, column=1, padx=10)
button_predict.grid(row=2, column=0, columnspan=2, pady=10)
label_result.grid(row=3, column=0, columnspan=2, pady=10)

# Start the GUI event loop
window.mainloop()