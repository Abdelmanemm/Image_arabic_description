from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration , BlipForQuestionAnswering
import torch

# Define Class
class ImageCaption:
    def __init__(self):
        """Initialize the models and processors for image captioning and VQA"""
        # Load caption model
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        # Load VQA model
        self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    def get_caption(self, image):
        """Generate a caption for the input image"""
        text = "a photo of"
        inputs = self.caption_processor(image, text, return_tensors="pt")
        caption = self.caption_model.generate(**inputs)
        # Fix: Correcting processor variable reference
        caption = self.caption_processor.decode(caption[0], skip_special_tokens=True)
        return caption

    def ask_question(self, image, question):
        """Answer a question related to the input image"""
        inputs = self.vqa_processor(image, question, return_tensors="pt")
        # Fix: Use proper method for VQA (not generate)
        output = self.vqa_model.generate(**inputs)
        # Decoding answer using processor
        ans = self.vqa_processor.decode(output[0], skip_special_tokens=True)
        return ans
