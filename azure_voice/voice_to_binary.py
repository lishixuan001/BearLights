import wave

w = wave.open("test.wav", "rb")
binary_data = w.readframes(w.getnframes())
w.close()

print(binary_data)