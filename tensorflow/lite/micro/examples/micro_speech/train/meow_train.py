import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

start_time = datetime.now()

# A comma-delimited list of the words you want to train for.
# The options are: yes,no,up,down,left,right,on,off,stop,go
# All the other words will be used to train an "unknown" label and silent
# audio data with no spoken words will be used to train a "silence" label.
WANTED_WORDS = "meow"

# The number of steps and learning rates can be specified as comma-separated
# lists to define the rate at each stage. For example,
# TRAINING_STEPS=12000,3000 and LEARNING_RATE=0.001,0.0001
# will run 12,000 training loops in total, with a rate of 0.001 for the first
# 8,000, and 0.0001 for the final 3,000.
TRAINING_STEPS = "12000,3000"
LEARNING_RATE = "0.001,0.0001"

# Calculate the total number of steps, which is used to identify the checkpoint
# file name.
TOTAL_STEPS = str(sum(map(lambda string: int(string), TRAINING_STEPS.split(","))))

# Print the configuration to confirm it
print("Training these words: %s" % WANTED_WORDS)
print("Training steps in each stage: %s" % TRAINING_STEPS)
print("Learning rate in each stage: %s" % LEARNING_RATE)
print("Total number of training steps: %s" % TOTAL_STEPS)

# Calculate the percentage of 'silence' and 'unknown' training samples required
# to ensure that we have equal number of samples for each label.
number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels + 2  # for 'silence' and 'unknown' label
equal_percentage_of_training_samples = int(100.0 / (number_of_total_labels))
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples

# Constants which are shared during training and inference
PREPROCESS = 'micro'
WINDOW_STRIDE = 20
MODEL_ARCHITECTURE = 'tiny_conv'  # Other options include: single_fc, conv,
# low_latency_conv, low_latency_svdf, tiny_embedding_conv

# Constants used during training only
VERBOSITY = 'WARN'
EVAL_STEP_INTERVAL = '1000'
SAVE_STEP_INTERVAL = '1000'

# Constants for training directories and filepaths
DATASET_DIR = '/tmp/samples/'
LOGS_DIR = 'logs/'
TRAIN_DIR = 'train/'  # for training checkpoints and other files.

# Constants for inference directories and filepaths

MODELS_DIR = 'models/'
if not os.path.exists(MODELS_DIR):
  os.mkdir(MODELS_DIR)
MODEL_TF = os.path.join(MODELS_DIR, 'meow.pb')
MODEL_TFLITE = os.path.join(MODELS_DIR, 'meow.tflite')
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, 'float_meow.tflite')
MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, 'meow.cc')
SAVED_MODEL = os.path.join(MODELS_DIR, 'saved_model')

QUANT_INPUT_MIN = 0.0
QUANT_INPUT_MAX = 26.0
QUANT_INPUT_RANGE = QUANT_INPUT_MAX - QUANT_INPUT_MIN
COMMAND_DIR = (
  '/Users/tennis/src/tflite-micro-fork/tensorflow/lite/micro/examples/micro_speech/tensorflow/tensorflow/examples'
  '/speech_commands')

print("Training the model (this may take quite a while)...")
os.system(f'python {COMMAND_DIR}/train.py \
        --data_url= "" \
        --data_dir={DATASET_DIR} \
        --wanted_words={WANTED_WORDS} \
        --silence_percentage={SILENT_PERCENTAGE} \
        --unknown_percentage={UNKNOWN_PERCENTAGE} \
        --preprocess={PREPROCESS} \
        --window_stride={WINDOW_STRIDE} \
        --model_architecture={MODEL_ARCHITECTURE} \
        --how_many_training_steps={TRAINING_STEPS} \
        --learning_rate={LEARNING_RATE} \
        --train_dir={TRAIN_DIR} \
        --summaries_dir={LOGS_DIR} \
        --verbosity={VERBOSITY} \
        --eval_step_interval={EVAL_STEP_INTERVAL} \
        --save_step_interval={SAVE_STEP_INTERVAL}'
          )

print("Freezing the model")

os.system(f'python {COMMAND_DIR}/freeze.py \
       --wanted_words={WANTED_WORDS} \
       --window_stride_ms={WINDOW_STRIDE} \
       --preprocess={PREPROCESS} \
       --model_architecture={MODEL_ARCHITECTURE} \
       --start_checkpoint={TRAIN_DIR}{MODEL_ARCHITECTURE}.ckpt-{TOTAL_STEPS} \
       --save_format=saved_model \
       --output_file={SAVED_MODEL}'
          )

SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

DATA_URL = ''
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

sys.path.append(COMMAND_DIR)
import input_data
import models

model_settings = models.prepare_model_settings(
  len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
  SAMPLE_RATE,
  CLIP_DURATION_MS,
  WINDOW_SIZE_MS,
  WINDOW_STRIDE,
  FEATURE_BIN_COUNT,
  PREPROCESS
)

audio_processor = input_data.AudioProcessor(
  DATA_URL,
  DATASET_DIR,
  SILENT_PERCENTAGE,
  UNKNOWN_PERCENTAGE,
  WANTED_WORDS.split(','),
  VALIDATION_PERCENTAGE,
  TESTING_PERCENTAGE,
  model_settings,
  LOGS_DIR
)


def representative_dataset_gen():
  for i in range(8):
    data, _ = audio_processor.get_data(1, i * 1, model_settings,
                                       BACKGROUND_FREQUENCY,
                                       BACKGROUND_VOLUME_RANGE,
                                       TIME_SHIFT_MS,
                                       'testing',
                                       sess)
    flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 1960)
    print(flattened_data)
    yield [flattened_data]


with tf.compat.v1.Session() as sess:
  float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  float_tflite_model = float_converter.convert()
  float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
  print("Float model is %d bytes" % float_tflite_model_size)

  converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8
  converter.representative_dataset = representative_dataset_gen
  tflite_model = converter.convert()
  tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
  print("Quantized model is %d bytes" % tflite_model_size)


# Helper function to run inference
def run_tflite_inference(tflite_model_path, model_type="Float"):
  # Load test data
  np.random.seed(0)  # set random seed for reproducible test results.
  with tf.compat.v1.Session() as sess:
    test_data, test_labels = audio_processor.get_data(
      -1, 0, model_settings, BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
      TIME_SHIFT_MS, 'testing', sess)
  test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(tflite_model_path,
                                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  # For quantized models, manually quantize the input data from float to integer
  if model_type == "Quantized":
    input_scale, input_zero_point = input_details["quantization"]
    test_data = test_data / input_scale + input_zero_point
    test_data = test_data.astype(input_details["dtype"])

  correct_predictions = 0
  for i in range(len(test_data)):
    interpreter.set_tensor(input_details["index"], test_data[i])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    top_prediction = output.argmax()
    correct_predictions += (top_prediction == test_labels[i])

  print('%s model accuracy is %f%% (Number of test samples=%d)' % (
    model_type, (correct_predictions * 100) / len(test_data), len(test_data)))


# Compute float model accuracy
run_tflite_inference(FLOAT_MODEL_TFLITE)

# Compute quantized model accuracy
run_tflite_inference(MODEL_TFLITE, model_type='Quantized')

os.system(f'xxd -i {MODEL_TFLITE} > {MODEL_TFLITE_MICRO}')
REPLACE_TEXT = MODEL_TFLITE.replace('/', '_').replace('.', '_')
# print(f"Replacing {REPLACE_TEXT} in {MODEL_TFLITE_MICRO}")
# os.system(f"sed -i 's/{REPLACE_TEXT}/g_model/g' {MODEL_TFLITE_MICRO}")

# Calculating the number of feature vectors (kFeatureCount)
kFeatureCount = 1 + ((CLIP_DURATION_MS - WINDOW_SIZE_MS) // WINDOW_STRIDE)
kMaxAudioSampleSize = int((SAMPLE_RATE / 1000) * WINDOW_SIZE_MS)

print(f"kAudioSampleFrequency = {SAMPLE_RATE}\n"
      f"kMaxAudioSampleSize = {kMaxAudioSampleSize}\n"
      f"kFeatureSize = {FEATURE_BIN_COUNT}\n"
      f"kFeatureCount = {kFeatureCount}\n"
      f"kFeatureElementCount (kFeatureSize * kFeatureCount) = {FEATURE_BIN_COUNT * kFeatureCount}\n"
      f"kFeatureStrideMs = {WINDOW_STRIDE}\n"
      f"kFeatureDurationMs = {WINDOW_SIZE_MS}\n")

# Mark the end time
end_time = datetime.now()

# Calculate the total execution time
execution_time = end_time - start_time

# Convert the execution time to hours and minutes
execution_hours, remainder = divmod(execution_time.seconds, 3600)
execution_minutes = remainder // 60

print(f"Total execution time: {execution_hours} hours and {execution_minutes} minutes")
