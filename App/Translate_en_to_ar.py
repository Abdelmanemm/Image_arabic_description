from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define Class
class Translate_en_to_ar:
    ''' Take an English text and translate it to Arabic using LLM Silma-9B '''
    def __init__(self,model_id="silma-ai/SILMA-9B-Instruct-v1.0"):
        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,)
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Define System Prompt
        self.system_prompt = {
            "role": "system", 
            "content": (
                "أنت نموذج مختص بترجمة الأوصاف القصيرة من اللغة الإنجليزية إلى اللغة العربية الفصحى. "
                "عليك أن تنتج الترجمة العربية فقط، وتحرص على أن تكون الترجمة صحيحة ودقيقة مع بعض المرونة لتبدو طبيعية في العربية."
            )
        }
    def get_translation(self,text):
        # user input message
        messages = [ self.system_prompt,
            {"role": "user", "content": f"Translate the following message: '{text}'"}
        ]
        # Tokenize input
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")
        # Generate the output
        outputs = self.model.generate(**input_ids, max_new_tokens=256)
        # Decode the output and clean it up
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Make sure to output only Arabic translation
        if "model" in output_text:
            output_text = output_text.split("model")[1].strip()
        return output_text
