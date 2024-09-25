from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Define Class
class ImageCaption:
    ''' Take an Image and return a Caption describing Image '''
    def __init__(self,model_id="Salesforce/blip-image-captioning-large"):
        # Load model
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
        # Load Proccessor
        self.processor =  BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        
    def get_caption(self,img):
        # Get input tensor
        inputs = self.processor(img,return_tensors="pt").to("cuda", torch.float16)
        # generate output 
        output = self.model.generate(**inputs)
        # decode output
        output = self.processor.decode(output[0], skip_special_tokens=True)
        return output
