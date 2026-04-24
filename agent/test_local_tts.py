import os
from tts import synthesize

print("Testing Local TTS Generation with Cross-Coded text...")
test_text = input("Enter the text you want to synthesize: ")

if not test_text.strip():
    test_text = "আমি আজকে meeting korbo because project er deadline tomorrow."

print(f"Generating audio for text: '{test_text}'")
audio_bytes = synthesize(test_text)

out_file = "test_output.mp3"
with open(out_file, "wb") as f:
    f.write(audio_bytes)

print(f"Success! Audio saved to '{os.path.abspath(out_file)}'")
print("You can double click this file or play it using your local media player.")
