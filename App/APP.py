from ImageCaption import ImageCaption
from TranslateText import TranslateText
from TextToAudio import TextToAudio
from QuestionGenerator import QuestionGenerator
import gradio as gr

# Define App Class 
class APP:
    def __init__(self):
        # Initialize all the required components
        self.translator = TranslateText()
        self.caption_model = ImageCaption()
        self.question_generator = QuestionGenerator()
        self.tts = TextToAudio()

    def generate_caption(self, image):
        # Step 1: Generate image caption
        caption = self.caption_model.get_caption(image)

        # Step 2: Translate caption to Arabic
        translated_caption = self.translator.get_translation(caption, en_ar=True)

        # Step 3: Generate Arabic speech from caption
        speech = self.tts.get_audio(translated_caption[0])

        # Step 4 : Generate questions from caption
        suggested_questions = self.question_generator.suggest_questions(caption)

        # Step 5: Translate suggested questions to Arabic
        translated_questions = []
        for question in suggested_questions:
            translated_questions.append(self.translator.get_translation(question, en_ar=True))

        return translated_caption[0], speech , translated_questions

    def answer_question(self, image, question):
        # Translate queston from Arabic to English
        question = self.translator.get_translation(question, ar_en=True)

        # Step 4: Answer the question based on the image
        answer = self.caption_model.ask_question(image, question)

        # Step 5: Translate the answer to Arabic
        translated_answer = self.translator.get_translation(answer, en_ar=True)
        return translated_answer[0]

    
# APP object which will be used in the interface 
app = APP()

# Creating Gradio interface to interact and use or pipeline in high level
# Function to generate the caption, speech, and questions
def generate_caption(image):
    try:
        # Get the translated caption, speech, and suggested questions
        translated_caption, speech, suggested_questions = app.generate_caption(image)

        # Clean and format questions
        questions_cleaned = [q[0] for q in suggested_questions]  # Extract each question from the list of lists

        # Ensure there are 4 questions, if less, fill with empty strings
        while len(questions_cleaned) < 4:
            questions_cleaned.append("")  # Fill with empty strings if fewer than 4 questions

        # Return caption text, audio file, and each question separately
        return translated_caption, speech, questions_cleaned[0], questions_cleaned[1], questions_cleaned[2], questions_cleaned[3]

    except Exception as e:
        # Return default values in case of error
        return "Error generating caption.", None, "Error: Couldn't generate question 1", "Error: Couldn't generate question 2", "Error: Couldn't generate question 3", "Error: Couldn't generate question 4"

# Function to answer the user question
def answer_question(image, user_question):
    try:
        answer = app.answer_question(image, user_question)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Function to populate the user question textbox
def set_user_question(question):
    return question

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 align='center'>Image Captioning and Question Answering App (Arabic)</h1>")

    with gr.Row():
        image_input = gr.Image(label="Upload an Image", type="pil")

    with gr.Row():
        generate_btn = gr.Button("Generate Caption")

    with gr.Row():
        caption_output = gr.Textbox(label="Caption", lines=2)
        audio_output = gr.Audio(label="Caption Audio")

    # Create buttons for each question
    with gr.Row():
        question_btn1 = gr.Button("Suggested Question 1")
        question_output1 = gr.Textbox(label="Suggested Question", lines=1)
        question_btn2 = gr.Button("Suggested Question 2")
        question_output2 = gr.Textbox(label="Suggested Question", lines=1)


    with gr.Row():
        user_question = gr.Textbox(label="Ask a Question (in Arabic)")

    with gr.Row():
        answer_btn = gr.Button("Get Answer")

    with gr.Row():
        answer_output = gr.Textbox(label="Answer (Arabic)", lines=2)

    # Function bindings
    generate_btn.click(
        generate_caption,
        inputs=[image_input],
        outputs=[caption_output, audio_output, question_output1, question_output2]
    )

    # Set user question when a button is clicked
    question_btn1.click(set_user_question, inputs=question_output1, outputs=user_question)
    question_btn2.click(set_user_question, inputs=question_output2, outputs=user_question)


    answer_btn.click(answer_question, inputs=[image_input, user_question], outputs=[answer_output])

# Launch the interface
demo.launch()
