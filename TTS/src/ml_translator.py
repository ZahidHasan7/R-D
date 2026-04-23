from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import warnings

# Suppress HuggingFace warnings for cleaner output
warnings.filterwarnings("ignore")

class MLTranslator:
    def __init__(self):
        print("Loading csebuetnlp/banglat5 Seq2Seq model... (This may take a moment)")
        model_name = "csebuetnlp/banglat5_nmt_en_bn"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text):
        """
        Translates remaining English context to Bengali.
        To prevent the NMT model from paraphrasing mostly-Bengali sentences,
        we only send the English segments to the model.
        """
        if not re.search(r'[a-zA-Z]{2,}', text):
            return text

        # If the sentence is mostly Bengali, don't translate the whole thing
        # Heuristic: if Bengali characters > 50% of the non-space characters
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if bengali_chars > english_chars and english_chars < 20:
            # Handle English words individually or as small chunks
            def _replace_en_chunk(match):
                en_text = match.group(0)
                # For very short English (1-2 words), ML translation might paraphrase too much
                # but for this specific model, we'll try it on the chunk
                try:
                    inputs = self.tokenizer(en_text, return_tensors="pt", padding=True, truncation=True)
                    outputs = self.model.generate(**inputs, max_length=50)
                    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                except:
                    return en_text

            # Regex for English words/chunks
            return re.sub(r'[a-zA-Z]{2,}(?:\s+[a-zA-Z]{2,})*', _replace_en_chunk, text)

        # If it's mostly English or long segments, translate the whole thing
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model.generate(**inputs, max_length=128)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
        except Exception as e:
            print(f"ML Translation error: {e}")
            return text
