import sys

from pydub import AudioSegment


def convert_wav(input_file, output_file):
  # Load the input file
  audio = AudioSegment.from_wav(input_file)

  # Set the target parameters
  target_sample_rate = 16000
  target_channels = 1
  target_bit_depth = 16
  target_duration_ms = 30  # Duration in milliseconds

  # Convert to mono if necessary
  if audio.channels != target_channels:
    audio = audio.set_channels(target_channels)

  # Resample
  audio = audio.set_frame_rate(target_sample_rate)

  # Change bit depth
  audio = audio.set_sample_width(target_bit_depth // 8)  # Pydub uses bytes for sample width

  # Trim or extend the audio to the target duration
  target_length = target_duration_ms * target_sample_rate // 1000  # Convert ms to samples
  audio = audio[:target_duration_ms]  # Trim to the first 30 ms

  # Check if the audio is shorter than the target duration and pad with silence if necessary
  current_length = len(audio)  # Length in milliseconds
  if current_length < target_duration_ms:
    silence_duration = target_duration_ms - current_length
    silence = AudioSegment.silent(duration=silence_duration, frame_rate=target_sample_rate)
    audio += silence  # Append silence to the end of the audio

  # Ensure the frame count is 480 (this should already be the case based on the calculations above)
  assert len(
    audio.get_array_of_samples()) == target_length, f"Unexpected number of frames: {len(audio.get_array_of_samples())}"

  # Export the processed audio
  audio.export(output_file, format="wav", parameters=["-ac", "1", "-sample_fmt", "s16", "-ar", "16000"])
  print(f"Output file {output_file} has been created with the target specifications.")


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python script.py <input_wav_file> <output_wav_file>")
    sys.exit(1)

  input_file = sys.argv[1]
  output_file = sys.argv[2]

  convert_wav(input_file, output_file)
