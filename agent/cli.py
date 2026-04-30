import os
import sys
import argparse

# Add project root to path
sys.path.append(os.getcwd())

from agent.tts import synthesize
from agent.asr import transcribe

def run_cli():
    print("====================================================")
    print("   Hybrid Voice Agent - Interactive CLI")
    print("====================================================")
    print("Available Commands:")
    print("  tts <text>  - Synthesize code-mixed text")
    print("  asr <path>  - Transcribe an audio file")
    print("  exit        - Quit the CLI")
    print("----------------------------------------------------\n")

    while True:
        try:
            line = input("Agent > ").strip()
            if not line:
                continue
                
            if line.lower() == "exit":
                break
                
            parts = line.split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd == "tts":
                if not arg:
                    print("[!] Please provide text to synthesize.")
                    continue
                print(f"[*] Synthesizing: {arg}")
                audio_bytes = synthesize(arg)
                out_path = "output_cli.wav"
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"[+] Audio saved to: {out_path}")
                
            elif cmd == "asr":
                if not arg:
                    print("[!] Please provide an audio file path.")
                    continue
                if not os.path.exists(arg):
                    print(f"[!] File not found: {arg}")
                    continue
                print(f"[*] Transcribing: {arg}")
                text = transcribe(arg)
                print(f"[+] Transcript: {text}")
                
            else:
                print(f"[!] Unknown command: {cmd}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[!] Error: {e}")

if __name__ == "__main__":
    run_cli()
