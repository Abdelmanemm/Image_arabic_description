# Automated Image Captioning and Q&A Pipeline

This project implements an automated pipeline that processes images, generates captions, translates captions to Arabic, and enables users to ask questions about the image. It utilizes multiple state-of-the-art machine learning models for image captioning, translation, and visual question answering (VQA). The system also provides audio output of the caption in Arabic, and the entire interface is powered by Gradio.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Notebook](#notebook)
- [Technologies Used](#technologies-used)
- [License](#license)

## Overview

The pipeline accepts an image as input, processes it through various AI models, and delivers both text and audio outputs in Arabic. It also allows the user to ask a question about the image and returns the answer, translated into Arabic.

### Key Components:
1. **Image Captioning:** Generates a descriptive caption for the input image.
2. **Translation:** Translates the generated caption from English to Arabic.
3. **Audio Output:** Converts the Arabic text to speech and plays the audio.
4. **Q&A:** Suggests questions related to the image, allowing the user to select or write their own. The system then answers the selected question.

## Features

- **Automated Caption Generation:** Uses [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) to generate captions for input images.
- **Arabic Translation:** Utilizes [facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) for caption translation.
- **Audio Playback:** Uses Google Text-to-Speech (gtts) to convert Arabic captions to audio files and plays the audio.
- **Visual Question Answering (VQA):** Allows the user to select or ask a question about the image, which is answered by [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) and translated to Arabic.
- **Interactive User Input:** Users can either select a suggested question or input their own.
- **Gradio Interface:** The pipeline is integrated with a user-friendly interface built using Gradio.

## Workflow

Here’s a step-by-step breakdown of the pipeline:

1. **Input Image**: User provides an image as input via the Gradio interface.
2. **Image Captioning**: The image is passed through [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large), which generates a caption for the image in English.
3. **Translation**: The generated caption is translated from English to Arabic using [facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt).
4. **Text and Audio Output**: The Arabic caption is displayed to the user, and the caption is converted to speech using Google Text-to-Speech (gtts). The audio file is played for the user.
5. **Question Suggestions**: The caption is sent to [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), which generates two suggested questions about the image. The user can select one of the suggested questions or input a custom question.
6. **VQA Processing**: The selected question is passed to [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base), which provides an answer. The answer is translated to Arabic.
7. **Answer Output**: The Arabic answer is displayed to the user.

## Installation

To set up this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Abdelmanemm/Image_arabic_description.git
    cd Image_arabic_description/App/
    ```
    
2. **Download necessary models**:
    Make sure to download the models used in this pipeline:
    - [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)
    - [facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)
    - gtts (Google Text-to-Speech)
    - [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
    - [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base)

3. **Run the pipeline**:
    You can run the main script to start the pipeline:
    ```bash
    python APP.py
    ```

## Usage

1. Provide an image input via the Gradio interface.
2. Wait for the system to generate a caption and translate it to Arabic.
3. Listen to the Arabic audio of the caption.
4. Choose a suggested question or input your own.
5. Receive the answer in Arabic.


## Notebook

A Jupyter notebook is provided for experimenting with the pipeline and understanding each step of the process in detail. You can find it [here](notebook/Image_Captioning_and_Question_Answering_App_(Arabic).ipynb).


## Technologies Used

- **[Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)**: For generating image captions.
- **[facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)**: For translating captions to Arabic.
- **gtts**: For converting Arabic text to speech.
- **[microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)**: For suggesting relevant questions.
- **[Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base)**: For answering questions related to the image.
- **Gradio**: For building the interactive user interface.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
