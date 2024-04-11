"""

"""
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score

COMMAND_DIR = (
  '/Users/tennis/'
  'esp/tensorflow/tensorflow/examples/speech_commands'
)

sys.path.append(COMMAND_DIR)
import input_data
import models

# A comma-delimited list of the words you want to train for.
# The options are: yes,no,up,down,left,right,on,off,stop,go
# All the other words will be used to train an "unknown" label and silent
# audio data with no spoken words will be used to train a "silence" label.
WANTED_WORDS = "meow"

# There are hidden categories used for training.
ALL_WORDS = list(set(input_data.prepare_words_list(WANTED_WORDS.split(','))))

# The number of steps and learning rates can be specified as comma-separated
# lists to define the rate at each stage. For example,
# TRAINING_STEPS=12000,3000 and LEARNING_RATE=0.001,0.0001
# will run 12,000 training loops in total, with a rate of 0.001 for the first
# 8,000, and 0.0001 for the final 3,000.
TRAINING_STEPS = "4000,2000"
LEARNING_RATE = "0.001,0.0001"

# Calculate the total number of steps, which is used to identify the checkpoint
# file name.
TOTAL_STEPS = str(sum(map(lambda string: int(string), TRAINING_STEPS.split(","))))

# Calculate the percentage of 'silence' and 'unknown' training samples required
# to ensure that we have equal number of samples for each label.
equal_percentage_of_training_samples = int(100.0 / len(ALL_WORDS))
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples

# Constants which are shared during training and inference
PREPROCESS = 'micro'
MODEL_ARCHITECTURE = 'tiny_conv'  # Other options include: single_fc, conv,
# low_latency_conv, low_latency_svdf, tiny_embedding_conv

# Constants used during training only
VERBOSITY = 'WARN'
EVAL_STEP_INTERVAL = '1000'
SAVE_STEP_INTERVAL = '1000'

# Constants for training directories and filepaths
DATASET_DIR = '/tmp/dataset-2024-04-05-04-14-29/'

datetime_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
PARENT_DIR = f"meow-training-{datetime_string}"

os.makedirs(PARENT_DIR)

LOGS_DIR = os.path.join(PARENT_DIR, 'logs/')
PLOTS_DIR = os.path.join(PARENT_DIR, 'plots/')
TRAIN_DIR = os.path.join(PARENT_DIR, 'train/')
MODELS_DIR = os.path.join(PARENT_DIR, 'models/')

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_TF = os.path.join(MODELS_DIR, 'meow.pb')
MODEL_TFLITE = os.path.join(MODELS_DIR, 'meow.tflite')
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, 'float_meow.tflite')
MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, 'meow.cc')
MODEL_TFLITE_MICRO_HEADER = os.path.join(MODELS_DIR, 'meow.h')
SAVED_MODEL = os.path.join(MODELS_DIR, 'saved_model')

QUANT_INPUT_MIN = 0.0
QUANT_INPUT_MAX = 26.0
QUANT_INPUT_RANGE = QUANT_INPUT_MAX - QUANT_INPUT_MIN

SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_STRIDE_MS = 20
WINDOW_SIZE_MS = 30
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

DATA_URL = ''
VALIDATION_PERCENTAGE = 25
TESTING_PERCENTAGE = 25

desired_samples = int(SAMPLE_RATE * CLIP_DURATION_MS / 1000)
window_size_samples = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
window_stride_samples = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)
length_minus_window = (desired_samples - window_size_samples)
# Adjusting the calculation here to more accurately reflect the handling of the last window
spectrogram_length = 1 + int(length_minus_window / window_stride_samples) if length_minus_window >= 0 else 0
TOTAL_FEATURE_DIMENSION = FEATURE_BIN_COUNT * spectrogram_length


def print_mean_std(sample, sample_index):
  mean = np.mean(sample)
  std = np.std(sample)
  print(f"Sample {sample_index}: Mean = {mean:.2f}, Std Dev = {std:.2f}")


def plot_results(predictions, true_labels):
  save_confusion_matrix(predictions,
                        true_labels,
                        ALL_WORDS,
                        PLOTS_DIR)

  metrics_dict = evaluate_multiclass_precision_recall(predictions,
                                                      true_labels,
                                                      ALL_WORDS,
                                                      plot_curves=True,
                                                      save_dir=PLOTS_DIR)

  plot_average_precision(metrics_dict, PLOTS_DIR)

  plot_precision_recall(metrics_dict, PLOTS_DIR)

  plot_f1_scores(metrics_dict, PLOTS_DIR)


def ensure_dir(directory):
  """Ensure the directory exists, and if not, create it."""
  if not os.path.exists(directory):
    os.makedirs(directory)


def plot_average_precision(metrics_dict, plots_dir):
  classes = list(metrics_dict.keys())
  average_precisions = [metrics['average_precision'] for metrics in metrics_dict.values()]

  plt.figure(figsize=(10, 6))
  sns.barplot(x=classes, y=average_precisions, palette='viridis')
  plt.xlabel('Classes')
  plt.ylabel('Average Precision (AP)')
  plt.title('Average Precision per Class')
  plt.xticks(rotation=45)
  plt.tight_layout()

  ensure_dir(plots_dir)
  plt.savefig(os.path.join(plots_dir, 'average_precision_per_class.png'))
  plt.close()


def plot_precision_recall(metrics_dict, plots_dir):
  classes = list(metrics_dict.keys())
  precisions = [metrics['precision'][0] for metrics in metrics_dict.values()]  # Taking the last precision value
  recalls = [metrics['recall'][0] for metrics in metrics_dict.values()]  # Taking the last recall value

  x = np.arange(len(classes))
  width = 0.35

  fig, ax = plt.subplots(figsize=(12, 6))
  bars1 = ax.bar(x - width / 2, precisions, width, label='Precision', color='skyblue')
  bars2 = ax.bar(x + width / 2, recalls, width, label='Recall', color='orange')

  ax.set_xlabel('Classes')
  ax.set_ylabel('Scores')
  ax.set_title('Precision and Recall per Class')
  ax.set_xticks(x)
  ax.set_xticklabels(classes, rotation=45)
  ax.legend()
  plt.tight_layout()

  ensure_dir(plots_dir)
  plt.savefig(os.path.join(plots_dir, 'precision_recall_per_class.png'))
  plt.close()


def plot_f1_scores(metrics_dict, plots_dir):
  classes = list(metrics_dict.keys())
  f1_scores = []
  for metrics in metrics_dict.values():
    precision = metrics['precision'][0]  # Taking the last precision value
    recall = metrics['recall'][0]  # Taking the last recall value
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1_score)

  plt.figure(figsize=(10, 6))
  sns.barplot(x=classes, y=f1_scores, palette='mako')
  plt.xlabel('Classes')
  plt.ylabel('F1 Score')
  plt.title('F1 Score per Class')
  plt.xticks(rotation=45)
  plt.tight_layout()

  ensure_dir(plots_dir)
  plt.savefig(os.path.join(plots_dir, 'f1_scores_per_class.png'))
  plt.close()


def log_precision(metrics_dict):
  # Example: Analyzing average precision across classes
  average_precisions = {class_id: metrics['average_precision'] for class_id, metrics in metrics_dict.items()}

  # Find the class with the lowest AP score
  worst_class_id = min(average_precisions, key=average_precisions.get)
  print(f"Class with the lowest AP score: {worst_class_id}, AP: {average_precisions[worst_class_id]}")


def write_c_source_files():
  os.system(f'xxd -i {MODEL_TFLITE} > {MODEL_TFLITE_MICRO}')
  REPLACE_TEXT = MODEL_TFLITE.replace('/', '_').replace('.', '_')
  # print(f"Replacing {REPLACE_TEXT} in {MODEL_TFLITE_MICRO}")
  # os.system(f"sed -i 's/{REPLACE_TEXT}/g_model/g' {MODEL_TFLITE_MICRO}")

  generate_model_header_file(
    CLIP_DURATION_MS,
    WINDOW_SIZE_MS,
    WINDOW_STRIDE_MS,
    SAMPLE_RATE,
    FEATURE_BIN_COUNT,
    WANTED_WORDS,
    MODEL_TFLITE_MICRO_HEADER
  )


def quantize(audio_processor, model_settings):
  with tf.compat.v1.Session() as sess:
    # float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    # float_tflite_model = float_converter.convert()
    # float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
    # print("Float model is %d bytes" % float_tflite_model_size)

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    def representative_dataset_gen():
      # Use could use 100 samples or 10% of the dataset, whichever is smaller - we assume 100 is smaller
      for i in range(100):
        data, _ = audio_processor.get_data(1, i * 1, model_settings,
                                           BACKGROUND_FREQUENCY,
                                           BACKGROUND_VOLUME_RANGE,
                                           TIME_SHIFT_MS,
                                           'testing',
                                           sess)
        flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, TOTAL_FEATURE_DIMENSION)
        yield [flattened_data]

    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
    print("Quantized model is %d bytes" % tflite_model_size)


def generate_model_header_file(CLIP_DURATION_MS, WINDOW_SIZE_MS, WINDOW_STRIDE_MS, SAMPLE_RATE, FEATURE_BIN_COUNT,
                               WANTED_WORDS, output_file_path):
  """
  Generates a C++ header file content for model configuration, ensuring all values are explicitly
  integers in Python before writing to the file. It includes #ifndef preprocessor directives
  to prevent double inclusion.

  Parameters:
  - CLIP_DURATION_MS: Clip duration in milliseconds.
  - WINDOW_SIZE_MS: Size of the window in milliseconds.
  - WINDOW_STRIDE_MS: Window stride in milliseconds.
  - SAMPLE_RATE: Sample rate of the audio in Hz.
  - FEATURE_BIN_COUNT: Number of feature bins.
  - WANTED_WORDS: A string of comma-separated words to detect.
  - output_file_path: The path to the file where the output will be written.
  """
  kFeatureCount = int(1 + ((CLIP_DURATION_MS - WINDOW_SIZE_MS) // WINDOW_STRIDE_MS))
  kMaxAudioSampleSize = int((SAMPLE_RATE / 1000) * WINDOW_SIZE_MS)
  kCategoryCount = int(len(ALL_WORDS))
  kFeatureSize = int(FEATURE_BIN_COUNT)
  kFeatureElementCount = int(FEATURE_BIN_COUNT * kFeatureCount)
  kFeatureStrideMs = int(WINDOW_STRIDE_MS)
  kFeatureDurationMs = int(WINDOW_SIZE_MS)
  kAudioSampleFrequency = int(SAMPLE_RATE)

  # Assuming list_to_c_array is adapted to handle integer conversion if necessary
  kCategoryLabelsArray = list_to_c_array(ALL_WORDS,
                                         "kCategoryLabels[kCategoryCount]",
                                         "constexpr const char*")

  header_guard = "MODEL_SETTINGS_H_"

  with open(output_file_path, 'w') as file:
    file_content = (
      f"#ifndef {header_guard}\n"
      f"#define {header_guard}\n"
      "\n"
      f"constexpr int kAudioSampleFrequency = {kAudioSampleFrequency};\n"
      f"constexpr int kMaxAudioSampleSize = {kMaxAudioSampleSize};\n"
      f"constexpr int kFeatureSize = {kFeatureSize};\n"
      f"constexpr int kFeatureCount = {kFeatureCount};\n"
      f"constexpr int kFeatureElementCount = {kFeatureElementCount}; // kFeatureSize * kFeatureCount\n"
      f"constexpr int kFeatureStrideMs = {kFeatureStrideMs};\n"
      f"constexpr int kFeatureDurationMs = {kFeatureDurationMs};\n"
      "\n"
      f"constexpr int kCategoryCount = {kCategoryCount};\n"
      f"{kCategoryLabelsArray}\n"
      "\n"
      "#endif // " + header_guard + "\n"
    )
    file.write(file_content)


def list_to_c_array(input_list, array_name="myArray", data_type="char*"):
  """
  Converts a list of strings to a C array declaration, removing underscores from the strings.

  Args:
  - input_list: List of strings to be included in the C array.
  - array_name: Name of the C array.
  - data_type: Data type of the C array.

  Returns:
  - A string representing the C array declaration.
  """
  # Removing underscores and quoting strings
  formatted_items = [f'"{item.replace("_", "")}"' for item in input_list]

  # Joining the formatted strings with commas
  array_items = ", ".join(formatted_items)

  # Creating the C array declaration string
  c_array_declaration = f"{data_type} {array_name} = {{{array_items}}};"

  return c_array_declaration


def evaluate_multiclass_precision_recall(predictions, true_labels, class_labels, plot_curves=False, save_dir=None):
  """
  Evaluates and optionally plots and saves the Precision-Recall curve for each class in a multiclass setting to a
  specified directory,
  with class names provided by the user.

  Parameters:
  - predictions: numpy array of shape (num_samples, num_classes) containing the prediction scores for each class.
  - true_labels: numpy array of shape (num_samples,) containing the true class labels.
  - class_labels: list of strings, containing labels for each class.
  - plot_curves: bool, if True, plot and save the Precision-Recall curve for each class.
  - save_dir: str, directory where plots should be saved. If None, plots are not saved even if plot_curves is True.

  Returns:
  - A dictionary containing precision, recall, and average precision for each class.
  """
  num_classes = predictions.shape[1]
  metrics_dict = {}

  for i in range(num_classes):
    # Prepare binary labels and scores for the current class
    binary_true_labels = (true_labels == i).astype(int)
    scores = predictions[:, i]

    # Compute precision, recall, and average precision
    precision, recall, _ = precision_recall_curve(binary_true_labels, scores)
    average_precision = average_precision_score(binary_true_labels, scores)

    # Store metrics
    class_name = class_labels[i] if i < len(class_labels) else f'Class {i}'
    metrics_dict[class_name] = {
      'precision': precision,
      'recall': recall,
      'average_precision': average_precision
    }

    # Optionally plot and save Precision-Recall curve
    if plot_curves and save_dir is not None:
      plt.figure()
      plt.step(recall, precision, where='post')
      plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.ylim([0.0, 1.05])
      plt.xlim([0.0, 1.0])
      plt.title(f'{class_name} Precision-Recall curve: AP={average_precision:0.2f}')

      # Check if save directory exists, create if not
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)

      # Save the figure
      plt.savefig(os.path.join(save_dir, f'{class_name.replace(" ", "_")}_Precision_Recall_Curve.png'))
      plt.close()

  return metrics_dict


def run_tflite_inference(audio_processor, model_settings, tflite_model_path, model_type="Float"):
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

  print("Input tensor details:", input_details)
  print("Output tensor details:", output_details)

  # For quantized models, manually quantize the input data from float to integer
  if model_type == "Quantized":
    input_scale, input_zero_point = input_details["quantization"]
    test_data = test_data / input_scale + input_zero_point
    test_data = test_data.astype(input_details["dtype"])

  TP = 0
  FP = 0
  FN = 0
  for i in range(len(test_data)):
    interpreter.set_tensor(input_details["index"], test_data[i])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    top_prediction = output.argmax()
    if top_prediction == test_labels[i]:
      TP += 1
    else:
      FP += 1
      FN += 1  # This assumes a binary classification. For multi-class, adjust accordingly.

  precision = TP / (TP + FP) if TP + FP > 0 else 0
  recall = TP / (TP + FN) if TP + FN > 0 else 0
  F1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

  print(f'{model_type} model Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {F1_score:.3f}'
        f' (Number of test samples={len(test_data)})')


def collect_model_predictions(audio_processor, model_settings, tflite_model_path, model_type="Float"):
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
  print(f"Input details: {input_details}")
  output_details = interpreter.get_output_details()[0]
  print(f"Output details: {output_details}")

  # Prepare arrays to store predictions and labels
  predictions = []
  labels = []

  # For quantized models, adjust the input data accordingly
  if model_type == "Quantized":
    input_scale, input_zero_point = input_details["quantization"]
    test_data = test_data / input_scale + input_zero_point
    test_data = test_data.astype(input_details["dtype"])

  # Collect predictions and true labels
  for i in range(len(test_data)):
    interpreter.set_tensor(input_details["index"], test_data[i])
    print_mean_std(test_data[i], i)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    predictions.append(output)
    labels.append(test_labels[i])

  return np.array(predictions), np.array(labels)


def save_confusion_matrix(predictions, true_labels, wanted_words, file_path):
  """
  Generates and saves a confusion matrix plot from predictions and true labels.

  Parameters:
  - predictions: The model's output probabilities for each class.
  - true_labels: The true labels for the data.
  - wanted_words: A comma-separated string of class names corresponding to the model's outputs.
  - file_path: The path to save the confusion matrix plot.
  """
  # Assuming predictions are probability scores, get the predicted class indices
  predicted_labels = np.argmax(predictions, axis=1)

  # Generate the confusion matrix
  cm = confusion_matrix(true_labels, predicted_labels)

  # Plot the confusion matrix
  plt.figure(figsize=(10, 7))
  sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=wanted_words, yticklabels=wanted_words)
  plt.title('Confusion Matrix')
  plt.ylabel('Actual Labels')
  plt.xlabel('Predicted Labels')

  # Save the plot to the specified file path
  plt.savefig(f"{file_path}/confusion_matrix.png")
  plt.close()  # Close the figure to free memory


def main():
  # Print the configuration to confirm it
  print("Training these words: %s" % WANTED_WORDS)
  print("Training steps in each stage: %s" % TRAINING_STEPS)
  print("Learning rate in each stage: %s" % LEARNING_RATE)
  print("Total number of training steps: %s" % TOTAL_STEPS)
  print("All words: %s" % ALL_WORDS)
  print("Total Feature Vector Dimension:", TOTAL_FEATURE_DIMENSION)

  print("Training the model (this will take quite a while)...")

  tensorboard_command = f'tensorboard --logdir {LOGS_DIR}'
  print("Follow using Tensorboard:\n\t", tensorboard_command)

  train_exit_status = os.system(f'python {COMMAND_DIR}/train.py '
                                f'--data_url= ""'
                                f' --data_dir={DATASET_DIR}'
                                f' --wanted_words={WANTED_WORDS}'
                                f' --silence_percentage={SILENT_PERCENTAGE}'
                                f' --unknown_percentage={UNKNOWN_PERCENTAGE}'
                                f' --preprocess={PREPROCESS}'
                                f' --window_stride={WINDOW_STRIDE_MS}'
                                f' --model_architecture={MODEL_ARCHITECTURE}'
                                f' --how_many_training_steps={TRAINING_STEPS}'
                                f' --learning_rate={LEARNING_RATE}'
                                f' --train_dir={TRAIN_DIR}'
                                f' --summaries_dir={LOGS_DIR}'
                                f' --verbosity={VERBOSITY}'
                                f' --eval_step_interval={EVAL_STEP_INTERVAL}'
                                f' --save_step_interval={SAVE_STEP_INTERVAL}'
                                f' --testing_percentage={TESTING_PERCENTAGE}'
                                f' --validation_percentage={VALIDATION_PERCENTAGE}')

  if train_exit_status != 0:
    print("Training failed")
    exit(train_exit_status)

  print("Freezing the model")

  freeze_exit_status = os.system(f'python {COMMAND_DIR}/freeze.py'
                                 f' --wanted_words={WANTED_WORDS}'
                                 f' --window_stride_ms={WINDOW_STRIDE_MS}'
                                 f' --preprocess={PREPROCESS}'
                                 f' --model_architecture={MODEL_ARCHITECTURE}'
                                 f' --start_checkpoint={TRAIN_DIR}{MODEL_ARCHITECTURE}.ckpt-{TOTAL_STEPS}'
                                 f' --save_format=saved_model'
                                 f' --output_file={SAVED_MODEL}'
                                 f' --how_many_training_steps={TRAINING_STEPS}')

  if freeze_exit_status != 0:
    print("Freezing failed")
    exit(freeze_exit_status)

  model_settings = models.prepare_model_settings(len(ALL_WORDS),
                                                 SAMPLE_RATE,
                                                 CLIP_DURATION_MS,
                                                 WINDOW_SIZE_MS,
                                                 WINDOW_STRIDE_MS,
                                                 FEATURE_BIN_COUNT,
                                                 PREPROCESS)

  audio_processor = input_data.AudioProcessor(DATA_URL,
                                              DATASET_DIR,
                                              SILENT_PERCENTAGE,
                                              UNKNOWN_PERCENTAGE,
                                              ALL_WORDS,
                                              VALIDATION_PERCENTAGE,
                                              TESTING_PERCENTAGE,
                                              model_settings,
                                              LOGS_DIR)

  quantize(audio_processor, model_settings)

  # print("Compute FLOAT model accuracy")
  # run_tflite_inference(audio_processor,
  #                      model_settings,
  #                      FLOAT_MODEL_TFLITE)

  print("Compute quantized model accuracy")
  run_tflite_inference(audio_processor,
                       model_settings,
                       MODEL_TFLITE,
                       model_type='Quantized')

  predictions, true_labels = collect_model_predictions(audio_processor,
                                                       model_settings,
                                                       MODEL_TFLITE,
                                                       model_type='Quantized')
  plot_results(predictions, true_labels)

  write_c_source_files()


if __name__ == '__main__':
  start_time = datetime.now()

  main()

  # Mark the end time
  end_time = datetime.now()

  # Calculate the total execution time
  execution_time = end_time - start_time

  # Convert the execution time to hours and minutes
  execution_hours, remainder = divmod(execution_time.seconds, 3600)
  execution_minutes = remainder // 60

  print(f"Total execution time: {execution_hours} hours and {execution_minutes} minutes")
