from ImageCaption import ImageCaption
from Translate_en_to_ar import Translate_en_to_ar
from TextToAudio import TextToAudio
import gradio as gr


class APP:
    def __init__(self):
        self.image_captioner  = ImageCaption()
        self.text_to_audio = TextToAudio()
        self.translator  = Translate_en_to_ar()
        
    def Run(self,image):
        caption = self.image_captioner.get_caption(image)
        translation = self.translator.get_translation(caption)
        audio = self.text_to_audio.get_audio(translation,'image_descripton.mp3')
        return audio
    

app = APP()

iface = gr.Interface(
    fn=app.Run,
    inputs=gr.Image(type="numpy"),  # Accepts an image
    outputs=gr.Audio(type="filepath"),  # Outputs an audio file
    title="Image Caption Translator",
    description="Upload an image to generate an Arabic audio description.",
)

# Launch the Gradio app
iface.launch()