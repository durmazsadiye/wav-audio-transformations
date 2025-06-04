import numpy as np
import wave
from scipy.fft import fft, ifft

def get_file_path():
    return '487726__lilmati__ticking-timer-30-sec.wav'

def load_wav(file_path):
    wf = wave.open(file_path, 'rb')
    n_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    frame_rate = wf.getframerate()
    n_frames = wf.getnframes()
    raw_data = wf.readframes(n_frames)
    audio = np.frombuffer(raw_data, dtype=np.int16)
    wf.close()

    
    if n_channels == 2:
        audio = audio.reshape(-1, 2)

    return audio, frame_rate, sample_width, n_channels

def save_wav(file_path, data, sample_rate, sample_width=2, n_channels=1):
    """Save audio data to a WAV file."""
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    print(f"Saved to {file_path}")

# Step 1: Open the file and read its data
file_path = get_file_path()
audio_np, orig_rate, sample_width, n_channels = load_wav(file_path)
print(f"Loaded file: {file_path}")
print(f"Sample rate: {orig_rate} Hz")
print(f"Channels: {n_channels}")
print(f"Sample width: {sample_width} bytes")
print(f"Audio shape: {audio_np.shape}")


# (Steps 2-3: Save with different sample rates)
sample_rates = [8000, 16000, 24000, 36000, 44000]
for rate in sample_rates:
    print(f"\nProcessing at {rate} samples/sec")
    # Save with modified sample rate to replicate playback effect
    save_wav(f"rate_{rate}_samples.wav", audio_np, rate, sample_width, n_channels)


# (Step 4: Very high sampling rates)
high_sample_rates = [80000, 160000]
for rate in high_sample_rates:
    print(f"\nProcessing at {rate} samples/sec (High Rate)")
    save_wav(f"rate_{rate}_samples.wav", audio_np, rate, sample_width, n_channels)


# (Step 5: Change the amplitude by multiplying with coefficients)
coeffs = [0.5, 2.0]
for coef in coeffs:
    print(f"\nProcessing with amplitude coefficient: {coef}")
    scaled = np.clip(audio_np * coef, -32768, 32767).astype(np.int16)
    save_wav(f"amplitude_{coef}.wav", scaled, orig_rate, sample_width, n_channels)


# (Step 6: Resampling by taking every other sample)
print("\nProcessing downsampled audio (every other sample)")
if n_channels == 2:
    downsampled = audio_np[::2, :]
else:
    downsampled = audio_np[::2]
save_wav("downsampled.wav", downsampled, orig_rate // 2, sample_width, n_channels)


# (Step 7: Apply FFT, remove high frequencies, and apply inverse FFT)
print("\nApplying FFT, removing high frequencies, and applying inverse FFT")

if n_channels == 2:
   
    left_channel = audio_np[:, 0]
    right_channel = audio_np[:, 1]
    
   
    fft_left = fft(left_channel)
    half_left = len(fft_left) // 2
    fft_left[half_left//2:half_left + half_left//2] = 0
    reconstructed_left = np.real(ifft(fft_left)).astype(np.int16)
    
   
    fft_right = fft(right_channel)
    half_right = len(fft_right) // 2
    fft_right[half_right//2:half_right + half_right//2] = 0
    reconstructed_right = np.real(ifft(fft_right)).astype(np.int16)
    
 
    reconstructed = np.column_stack((reconstructed_left, reconstructed_right))
else:
   
    fft_data = fft(audio_np)
    half = len(fft_data) // 2
    fft_data[half//2:half + half//2] = 0
    reconstructed = np.real(ifft(fft_data)).astype(np.int16)

save_wav("low_pass_filtered.wav", reconstructed, orig_rate, sample_width, n_channels)

print("\nTüm işlemler tamamlandı. Dosyaları herhangi bir ses oynatıcı ile çalabilirsiniz.")
print("All processing complete. You can play the files with any audio player.")