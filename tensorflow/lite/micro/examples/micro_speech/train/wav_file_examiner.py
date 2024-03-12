import sys
import wave


def analyze_wav(file_path):
  with wave.open(file_path, 'rb') as wav_file:
    # Extract basic parameters
    n_channels = wav_file.getnchannels()  # Number of channels (1 for mono, 2 for stereo)
    sample_width = wav_file.getsampwidth()  # Sample width in bytes
    frame_rate = wav_file.getframerate()  # Sample rate
    n_frames = wav_file.getnframes()  # Number of audio frames
    comp_type = wav_file.getcomptype()  # Compression type ('NONE' for PCM)
    comp_name = wav_file.getcompname()  # Human-readable compression type description

    # Ensure 16-bit signed PCM data, mono channel
    assert sample_width == 2, "The sample width must be 2 bytes (16 bits) for 16-bit PCM data."
    assert n_channels == 1, "The file must be mono (single channel)."

    print(f"Sample Rate: {frame_rate} Hz")
    print(f"Channels: {n_channels}")
    print(f"Sample Width: {sample_width * 8} bits")
    print(f"Number of Frames: {n_frames}")
    print(f"Compression Type: {comp_type} ({comp_name})")


# Define your window frame and stride here as needed for further processing
window_frame = 1024  # Example frame size
window_stride = 512  # Example stride size

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_wav_file>")
  else:
    file_path = sys.argv[1]
    analyze_wav(file_path)
