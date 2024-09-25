from ImageCaption import ImageCaption
from Translate_en_to_ar import Translate_en_to_ar
from TextToAudio import TextToAudio
import gradio as gr

# Define App Class 
class APP:
    ''' Take an Image an return speech descibe image in arabic 
        When Class object is created it intialize all the 3 classes 
        Run which is our pipeline,take an image and return audio '''
    def __init__(self):
        self.image_captioner  = ImageCaption()
        self.text_to_audio = TextToAudio()
        self.translator  = Translate_en_to_ar()
        
    def Run(self,image):
        caption = self.image_captioner.get_caption(image)
        translation = self.translator.get_translation(caption)
        audio = self.text_to_audio.get_audio(translation,'image_descripton.mp3')
        return audio
    
# APP object which will be used in the interface 
app = APP()

# Creating Gradio interface to interact and use or pipeline in high level
iface = gr.Interface(
    fn=app.Run,
    inputs=gr.Image(type="numpy"),  
    outputs=gr.Audio(type="filepath"),  
    title="Image Caption Translator",
    description="Upload an image to generate an Arabic audio description.",
)

# Launch the Gradio app
iface.launch()
