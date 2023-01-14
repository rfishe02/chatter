import io
import base64

# Import libraries for Whisper speech to text.
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Import libraries for Blenderbot text generation.
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
blender_tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
blender_model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")

# Import libraries for text to speech.
import g2p_en
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd
tts_models, tts_cfg, tts_task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
tts_model = tts_models[0]
TTSHubInterface.update_cfg_with_data_cfg(tts_cfg, tts_task.data_cfg)
tts_generator = tts_task.build_generator(tts_models, tts_cfg)

# Import libraries for Flask application.
from flask import Flask, render_template, request, make_response, jsonify
from flask_cors import CORS
app = Flask(__name__)
cors = CORS(app, origins=["http://localhost:3000"])

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

    output["speaker"] = audio_to_text(file)
    output["response"] = generate_text_reply(output["speaker"])

    # Get wav file and encode to base64 since we're sending it over json, along with text data.
    tts_wav, tts_rate = text_to_audio(output["response"])

    audio_buffer = io.BytesIO()
    torch.save(tts_wav, audio_buffer)
    audio_buffer.seek(0)
    #output["response_base64_wav"] = base64.b64encode(audio_buffer.read())
    audio_buffer.close()

    response = make_response(jsonify(output), 200,)
    response.headers["Content-Type"] = "application/json"
    return response

def audio_to_text(audio_file):

    # We have to save the file to disk in order to load it with soundfile. Soundfile creates a numpy array in a specific format for the processor.
    # This approach may only suitable for a single user.
    audio_file.save('spoken.wav')
    sf_data, sf_samplerate = sf.read('spoken.wav')

    # TODO: Find a way to do this in memory.
    # Probably a good in memory solution, but this doesn't seem to be working on Mac M1:
    # MemoryError: Cannot allocate write+execute memory for ffi.callback(). You might be running on a system that prevents this. For more information, see https://cffi.readthedocs.io/en/latest/using.html#callbacks
    #file_buffer = io.BytesIO()
    #audio_file.save(file_buffer)
    #file_buffer.seek(0)
    #sf_data, sf_samplerate = sf.read(file_buffer)

    # Process the array & get the input features for the model.
    whisper_inputs = whisper_processor(sf_data, return_tensors="pt").input_features 

    # Use the input features to generate ids from the model, then use the processor to decode these ids.
    whisper_generated_ids = whisper_model.generate(inputs=whisper_inputs)
    whisper_transcription = whisper_processor.batch_decode(whisper_generated_ids, skip_special_tokens=True)[0]

    return whisper_transcription

def generate_text_reply(UTTERANCE):

    blender_inputs = blender_tokenizer([UTTERANCE], return_tensors="pt")
    blender_reply_ids = blender_model.generate(**blender_inputs)
    blender_output = blender_tokenizer.batch_decode(blender_reply_ids, skip_special_tokens=True)[0]

    return blender_output

def text_to_audio(generated_reply):

    tts_sample = TTSHubInterface.get_model_input(tts_task, generated_reply)
    tts_wav, tts_rate = TTSHubInterface.get_prediction(tts_task, tts_model, tts_generator, tts_sample)

    return tts_wav, tts_rate

if __name__ == "__main__":
    app.run(host='localhost', port=5050, debug=True)