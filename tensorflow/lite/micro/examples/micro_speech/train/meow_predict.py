import time

import librosa  # You might need to install librosa for audio feature extraction
import numpy as np
import sounddevice as sd
import tensorflow.lite as tflite

# Parameters
model_path = './2024-03-17_09-58-52/models/meow.tflite'
sample_rate = 16000  # Adjust this to the rate your model expects
duration = 1  # Duration of recording, in seconds
channels = 1  # Mono audio

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def extract_features(audio_data, sample_rate):
  # Correctly call the mfcc function with named parameters
  features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
  return features


def preprocess_audio(features, expected_shape):
  # Ensure the correct number of MFCC features
  assert features.shape[0] == expected_shape[1], "Number of MFCC features does not match model's expectation."

  # Adjust the time steps to match the expected shape
  if features.shape[1] > expected_shape[2]:
    # Trim the excess
    features = features[:, :expected_shape[2]]
  elif features.shape[1] < expected_shape[2]:
    # Pad with zeros
    padding_width = expected_shape[2] - features.shape[1]
    features = np.pad(features, pad_width=((0, 0), (0, padding_width)), mode='constant')

  # Reshape to match the expected input shape (including batch size)
  features = features.reshape(expected_shape)
  return features


def predict(raw_audio_data, sample_rate):
  # Extract features and preprocess as before
  features = extract_features(raw_audio_data, sample_rate)
  preprocessed_audio_data = preprocess_audio(features, (1, 40, 49))  # This shape might need adjustment

  # Retrieve the model's input tensor quantization parameters and expected shape
  input_scale, input_zero_point = input_details[0]['quantization']
  expected_input_shape = input_details[0]['shape']

  print("Model's expected input shape:", expected_input_shape)

  # Ensure the preprocessed data matches the model's expected input shape
  # Adjust this line based on the model's requirements
  audio_data_ready = preprocessed_audio_data.reshape(expected_input_shape).astype(np.float32)

  # Quantize the input data to match the model's input type
  audio_data_quantized = np.round(audio_data_ready / input_scale + input_zero_point).astype(np.int8)

  # Set the quantized tensor as the model input
  interpreter.set_tensor(input_details[0]['index'], audio_data_quantized)

  # Run inference
  interpreter.invoke()

  # Get the prediction result
  output_data = interpreter.get_tensor(output_details[0]['index'])
  predicted_class = np.argmax(output_data)
  return predicted_class


def listen_and_predict():
  while True:
    print("Listening...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()  # Wait until recording is finished

    # Update this line to include 'sample_rate' as the second argument to 'predict'
    predicted_class = predict(audio_data.flatten(), sample_rate)

    if predicted_class == 0:
      print("Meow detected")
    elif predicted_class == 1:
      print("Silence detected")
    else:
      print("Unknown sound detected")

    time.sleep(5)  # Wait for 5 seconds before the next recording


if __name__ == "__main__":
  listen_and_predict()
