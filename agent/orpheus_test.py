import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf
import os
import time

# Model IDs
MODEL_ID = "ehzawad/orpheus-bangla-emotional-tts"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"

def test_orpheus():
    print(f"[*] Loading Tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print(f"[*] Loading Orpheus Model (CPU Mode)...")
    # Using bfloat16 to fit 3B model (6GB) into 8.4GB available RAM
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    print(f"[+] Model loaded in {time.time() - start_time:.2f}s")

    print(f"[*] Loading SNAC Model...")
    snac_model = SNAC.from_pretrained(SNAC_MODEL_ID).to("cpu")

    # ChatML/Llama-3 template usage for Orpheus with min_new_tokens
    messages = [
        {"role": "user", "content": "<happy>হ্যালো! আমি অরফিয়াস। আমি অনেক খুশি যে আপনার সাথে কথা বলছি।</happy>"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "<|audio|>"
    
    print(f"[*] Formatted Prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    start_gen = time.time()
    with torch.no_grad():
        gen_tokens = model.generate(
            **inputs,
            max_new_tokens=1024,
            min_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    print(f"[+] Tokens generated in {time.time() - start_gen:.2f}s")

    # The model outputs tokens that need to be decoded into SNAC codes
    # For Orpheus, the generated tokens contain the speech codes
    # We need to extract them. Usually, they start after the prompt.
    generated_ids = gen_tokens[0][inputs.input_ids.shape[1]:]

    print(f"[*] Generated {len(generated_ids)} tokens.")
    print(f"[*] Raw Token IDs (first 20): {generated_ids[:20].tolist()}")
    print(f"[*] Max Token ID: {generated_ids.max() if len(generated_ids) > 0 else 'N/A'}")

    # Extraction Logic for Orpheus (Llama-based)
    # The speech tokens are range [128256, 128256 + 28672]
    # We strictly filter out any tokens outside this specific SNAC range
    SPEECH_START_ID = 128256
    SPEECH_END_ID = 128256 + 28672 # 156928
    
    speech_tokens = [t.item() for t in generated_ids if SPEECH_START_ID <= t < SPEECH_END_ID]
    
    if not speech_tokens:
        print("[!] No valid speech tokens found in the output.")
        # Print what was found instead
        all_above = [t.item() for t in generated_ids if t >= SPEECH_START_ID]
        print(f"[*] Tokens above {SPEECH_START_ID}: {all_above}")
        return

    print(f"[*] Valid speech tokens found: {speech_tokens[:50]} ...")
    print(f"[*] Extracting SNAC codes from {len(speech_tokens)} valid tokens...")
    
    # Trim to multiple of 7 (SNAC 24kHz frame structure)
    num_frames = len(speech_tokens) // 7
    if num_frames == 0:
        print("[!] Not enough tokens to form a single frame.")
        return
        
    valid_token_count = num_frames * 7
    speech_tokens = speech_tokens[:valid_token_count]
        
    # Reconstruction
    l1, l2, l3 = [], [], []
    for i in range(num_frames):
        chunk = speech_tokens[i*7 : (i+1)*7]
        # L1 (0-4095): vocab[128256:132352]
        l1.append(chunk[0] - SPEECH_START_ID)
        # L2 (0-8191): vocab[132352:140544]
        l2.append([chunk[1] - (SPEECH_START_ID + 4096), chunk[2] - (SPEECH_START_ID + 4096)])
        # L3 (0-16383): vocab[140544:156928]
        l3.append([
            chunk[3] - (SPEECH_START_ID + 4096 + 8192),
            chunk[4] - (SPEECH_START_ID + 4096 + 8192),
            chunk[5] - (SPEECH_START_ID + 4096 + 8192),
            chunk[6] - (SPEECH_START_ID + 4096 + 8192)
        ])

    # Convert to Tensors for SNAC decoder
    codes = [
        torch.tensor(l1).unsqueeze(0),
        torch.tensor(l2).flatten().unsqueeze(0),
        torch.tensor(l3).flatten().unsqueeze(0)
    ]
    
    print(f"[*] Decoding SNAC codes to audio...")
    with torch.no_grad():
        audio_waveform = snac_model.decode(codes)
        
    audio_np = audio_waveform.cpu().numpy().flatten()
    
    output_path = "test_output/orpheus_emotional.wav"
    sf.write(output_path, audio_np, 24000)
    print(f"[+] Success! Emotional audio saved to {output_path}")
    print("Verification: Model loading and SNAC decoding confirmed.")

if __name__ == "__main__":
    if not os.path.exists("test_output"):
        os.makedirs("test_output")
    try:
        test_orpheus()
    except Exception as e:
        print(f"[!] Error: {e}")
