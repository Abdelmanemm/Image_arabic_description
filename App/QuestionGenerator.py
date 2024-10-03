import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define class
class QuestionGenerator:
    def __init__(self):
        # Set seed to 0
        torch.random.manual_seed(0)
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    def suggest_questions(self,caption):
        # Define system prompt and pass image caption to generate questions based on it
        messages = [
            {"role": "system", "content": "You are a model designed to generate thoughtful and precise questions based on image captions. Your task is to generate two questions related to the image described by the caption.The questions should ask about very simple, specific details that a question-answering model can respond to in one or two words. Keep these two questions short and straightforward to avoid confusion.Output only the questions."},
            {"role": "user", "content": "Here is an image caption: 'A person riding a red bicycle in a busy city street.'"},
            {"role": "assistant", "content": 'What color is the bicycle? \n Is the person wearing a helmet?\n'
            },
            {"role": "user", "content": "Here is an image caption: 'A dog playing in a garden with a ball.'"},
            {"role": "assistant", "content": 'What color is the dog?\n Are there any trees in the garden?\n'
            },
            {"role": "user", "content": f"Here is an image caption: '{caption}'"},
        ]
        # define pipeline
        pipe = pipeline(
            "text-generation",
            model= self.model,
            tokenizer= self.tokenizer,
        )
        # define generation arguments
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        # Generate the output
        output = pipe(messages, **generation_args)
        # return the generated questions
        return output[0]['generated_text'].split('\n')
