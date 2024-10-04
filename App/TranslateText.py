from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# Define Class
class TranslateText:
    def __init__(self):
        # Intialize model and tokenizer
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    def get_translation(self,text,ar_en=False,en_ar=False):
        if ar_en:
            # arabic text
            ar_text = text
            # translate Arabic to English
            self.tokenizer.src_lang = "ar_AR"
            encoded_ar = self.tokenizer(ar_text, return_tensors="pt")
            generated_tokens = self.model.generate(
                **encoded_ar,
                forced_bos_token_id= self.tokenizer.lang_code_to_id["en_XX"]
            )
            # return translation
            return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        else:
            # english text
            en_text = text
            # translate english to arabic
            self.tokenizer.src_lang = "en_XX"
            encoded_ar = self.tokenizer(en_text, return_tensors="pt")
            generated_tokens = self.model.generate(
                **encoded_ar,
                forced_bos_token_id= self.tokenizer.lang_code_to_id["ar_AR"]
            )
            # return translation
            return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
