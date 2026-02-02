#!/usr/bin/env python3
import sounddevice as sd
import numpy as np

print("Testing sounddevice blocking mode...")
try:
    stream = sd.InputStream(samplerate=16000, channels=1, dtype=np.float32, blocksize=1024)
    print("Stream created")
    stream.start()
    print("Stream started!")
    data, overflow = stream.read(1024)
    print(f"Read {len(data)} samples")
    stream.stop()
    stream.close()
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
