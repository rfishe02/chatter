import os
import io
import base64
import json
import sqlite3
from dotenv import load_dotenv

load_dotenv()

# Import libraries for speech to text.
import torch
import torchaudio
import soundfile as sf
from transformers import AutoProcessor, WhisperForConditionalGeneration
whisper_processor = AutoProcessor.from_pretrained("openai/whisper-small.en")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")

# Import libraries for text generation. 
chat_model = None
chat_tokenizer = None
client = None
if os.getenv('TEXT_GENERATION') == 'local':
    from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
    chat_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    chat_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
elif os.getenv('TEXT_GENERATION') == 'openai':
    from openai import OpenAI
    client = OpenAI(
        organization = os.getenv('OPENAI_ORG_ID'),
        api_key = os.getenv('OPENAI_API_KEY')
    )

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

# Import libraries for sentiment analysis.
import nltk 
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()

# Import libraries for Flask application.
from flask import Flask, render_template, request, make_response, jsonify
from flask_cors import CORS
app = Flask(__name__)
cors = CORS(app, origins=["http://localhost:3000"])

# Serving the index.html file from the templates folder.
@app.route("/")
def hello_world():
    return render_template("index.html") # TODO: Remove?

# Define a route that the end user will interact with.
@app.route("/reply", methods=['POST'])
def reply():

    output = {}

    file = request.files['converted']
    asr = audio_to_text(file)

    messages = get_recent_messages()
    messages = get_system_context(messages)
    messages.append({'role':'user', 'content': asr})

    txt_reply = ''
    if os.getenv('TEXT_GENERATION') == 'local':
        conversation_string = create_conversation_string(messages)
        txt_reply = generate_from_local(conversation_string)

    elif os.getenv('TEXT_GENERATION') == 'openai':
        txt_reply = generate_from_openai(messages)

    save_to_db(asr,txt_reply)

    tts_wav, tts_rate = text_to_audio(txt_reply)

    # Get wav file and encode to base64 since we're sending it over json along with text data.
    base64_tts_reply = encode_audio_base64(tts_wav, tts_rate)

    reply_sentiment = analyze_sentiment(txt_reply)

    output["txt_asr"] = asr 
    output["txt_reply"] = txt_reply 
    output["base64_wav"] = base64_tts_reply
    output["reply_sentiment"] = reply_sentiment

    response = make_response(output, 200,)
    response.headers["Content-Type"] = "application/json"
    return response

def get_recent_messages():
    messages = []
    sql = 'SELECT * FROM userchatlogs ORDER BY ROWID DESC LIMIT 10'

    con = sqlite3.connect("./chatter.db")
    cur = con.cursor()
    res = cur.execute(sql)
    res_set = res.fetchall()

    for row in res_set: 
        user_message = {}
        user_message['role'] = 'user'
        user_message['content'] = row[2]

        bot_message = {}
        bot_message['role'] = 'assistant'
        bot_message['content'] = row[3]

        messages.insert(0,bot_message)
        messages.insert(0,user_message)

    con.close()
    return messages

def get_system_context(messages):

    messages.insert(0,{"role": "system", "content": "You are a helpful assistant."})

    return messages

def create_conversation_string(messages):
    conversation_array = []
    conversation_string = ''

    for message in messages:
        conversation_array.append(message['content'])

    start_index = max(0, len(conversation_array) - 5)
    conversation_string = '\n'.join(conversation_array[start_index:])
    return conversation_string

def audio_to_text(audio_file):

    # We can save the file to disk in order to load it with soundfile. Soundfile creates a numpy array in a specific format for the processor.
    #audio_file.save('spoken.wav')
    #sf_data, sf_samplerate = sf.read('spoken.wav')

    # This is an in-memory solution that once had issues on Mac M1 (https://github.com/bastibe/python-soundfile/issues/331), but it appears to work now.
    file_buffer = io.BytesIO()
    audio_file.save(file_buffer)
    file_buffer.seek(0)
    sf_data, sf_samplerate = sf.read(file_buffer)

    # Process the array & get the input features for the model.
    whisper_inputs = whisper_processor(sf_data, return_tensors="pt")
    input_features = whisper_inputs.input_features 

    # Use the input features to generate ids from the model, then use the processor to decode these ids.
    whisper_generated_ids = whisper_model.generate(input_features=input_features)
    whisper_transcription = whisper_processor.batch_decode(whisper_generated_ids, skip_special_tokens=True)[0]

    return whisper_transcription

def save_to_db(user_message,bot_reply):

    con = sqlite3.connect("./chatter.db")
    cur = con.cursor()
    cur.execute('INSERT INTO userchatlogs(user_message,bot_response) VALUES(?,?)',(user_message,bot_reply,))
    con.commit()
    con.close()

def generate_from_local(conversation):
    reply_text = ''

    if len(conversation) > 0:
        inputs = chat_tokenizer([conversation], return_tensors="pt")
        reply_ids = chat_model.generate(**inputs)
        gen_reply_text = chat_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)

        reply_text = gen_reply_text[0]
    
    return reply_text

def generate_from_openai(messages):
    reply_text = ''

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    reply_text = completion.choices[0].message.content

    return reply_text

def text_to_audio(generated_reply):
    tts_wav, tts_rate = None, 0

    if len(generated_reply) > 0:
        tts_sample = TTSHubInterface.get_model_input(tts_task, generated_reply)
        tts_wav, tts_rate = TTSHubInterface.get_prediction(tts_task, tts_model, tts_generator, tts_sample)

    return tts_wav, tts_rate

def analyze_sentiment(txt_reply):
    reply_sentiment = 'neutral'
    
    if len(txt_reply) > 0:
        sentiment_dict = vader.polarity_scores(txt_reply)
    
        if sentiment_dict['compound'] >= 0.05 :
            reply_sentiment = 'positive'
        elif sentiment_dict['compound'] <= - 0.05 :
            reply_sentiment = 'negative'

    return reply_sentiment

def encode_audio_base64(tts_wav, tts_rate):
    base64_tts_reply = ''

    if tts_wav is not None:
        audio_buffer = io.BytesIO()
        tts_wav_2d_tensor = tts_wav.reshape(1,tts_wav.size()[0])
        torchaudio.save(audio_buffer, tts_wav_2d_tensor, tts_rate, format="wav")
        audio_buffer.seek(0)
        base64_tts_reply = base64.b64encode(audio_buffer.read()).decode('ASCII')
        audio_buffer.close()

    return base64_tts_reply

if __name__ == "__main__":

    # TODO: Possibly do table setup elsewhere.
    # TODO: Set up session based conversations?

    con = sqlite3.connect("./chatter.db")
    cur = con.cursor()
    res = cur.execute("SELECT * FROM sqlite_master WHERE type='table' AND name IN('userchatlogs')")
    if res.fetchone() is None:
        sql = '''
        CREATE TABLE userchatlogs(
	    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	    user_message TEXT NOT NULL,
        bot_response TEXT NOT NULL,
        reply_context TEXT
        );
        '''
        cur.executescript(sql)
    con.close()

    app.run(host='localhost', port=5050, debug=True)