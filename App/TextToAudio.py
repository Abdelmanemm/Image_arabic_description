from gtts import gTTS

# Define Class
class TextToAudio:
    ''' Take an Arabic text and return speech of text  '''
    def __init__(self):
        pass
    
    def get_audio(self,text,output_file_name="output.mp3"):
        try:
            # Generate the speech 
            tts = gTTS(text=text, lang='ar')
            
            # Save the audio to an mp3 file
            tts.save(output_file_name)
            
            # Return the file path
            return output_file_name
        
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None
