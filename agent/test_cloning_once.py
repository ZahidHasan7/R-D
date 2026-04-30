import sys, os
# Add current directory to path
sys.path.append(os.getcwd())
# Add TTS directory for internal imports
sys.path.append(os.path.abspath('TTS'))

from agent.tts import synthesize

test_text = "Hello! হ্যালো Zahid বলছি from OpenAI। Actually আজ সকাল 10:30 AM এ আমাদের multilingual Bangla-English TTS system এর final deployment ছিল।"
print(f"Synthesizing: {test_text}")

audio_bytes = synthesize(test_text)
if audio_bytes:
    with open("final_cloning_test.wav", "wb") as f:
        f.write(audio_bytes)
    print("Success! Audio saved to 'final_cloning_test.wav'")
else:
    print("Failed to generate audio.")
