import tensorflow as tf

# Path to the TensorFlow Lite model file
tflite_model_path = '/Users/tennis/src/tflite-micro-fork/tensorflow/lite/micro/examples/micro_speech/train/models/meow.tflite'

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get information about input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input and output details, which includes the shape.
print("Input details:", input_details)
print("Output details:", output_details)

# Specifically for the output details, you can get the output shape.
output_shape = output_details[0]['shape']
print("Output shape:", output_shape)

# The number of output categories can typically be derived from the last dimension of the output shape
num_output_categories = output_shape[-1]
print("Number of output categories:", num_output_categories)
