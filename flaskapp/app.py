from flask import Flask, render_template, request, make_response, jsonify
from flask_cors import CORS

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import torch
import numpy as np

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

# Define Flask application.
app = Flask(__name__)
cors = CORS(app, origins=["http://localhost:3000"])

# Define tools.

# Serving the index.html file from the templates folder.
@app.route("/")
def hello_world():
    return render_template("index.html")

# Define a route that the end user will interact with.
@app.route("/reply", methods=['POST'])
def reply():

    output = {"message":""}

    file = request.files['converted']
    print('File from the POST request is: {}'.format(file))

    output["message"] = audio_to_text(file)
    response = make_response(jsonify(output), 200,)
    response.headers["Content-Type"] = "application/json"
    return response

def audio_to_text(file):

    # We have to save the file to disk in order to load it with soundfile. Soundfile creates a numpy array in a specific format for the processor.
    file.save('./flaskapp/spoken.wav')
    data, samplerate = sf.read('./flaskapp/spoken.wav')

    # Process the array & get the input features for the model.
    inputs = processor(data, return_tensors="pt")
    input_features = inputs.input_features 

    # Use the input features to generate ids from the model, then use the processor to decode these ids.
    generated_ids = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return transcription

def generate_text_reply():
    return "wow2"

def text_to_audio():
    return "wow3"

if __name__ == "__main__":
    app.run(host='localhost', port=5050, debug=True)