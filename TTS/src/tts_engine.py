from gtts import gTTS
import os

class TTSEngine:
    def __init__(self, lang='bn'):
        self.lang = lang

    def generate_audio(self, text, output_path):
        """
        Generates audio for the given text and saves it to output_path.
        """
        try:
            # Keep commas for natural shorter pauses unless they cause issues.
            clean_text = text.replace('，', ',')
            
            tts = gTTS(text=clean_text, lang=self.lang)
            tts.save(output_path)
            return True
        except Exception as e:
            print(f"Error generating audio for '{text}': {e}")
            return False

if __name__ == "__main__":
    engine = TTSEngine()
    engine.generate_audio("৫ কেজি চাল", "test_bangla.mp3")
    print("Test audio generated: test_bangla.mp3")
