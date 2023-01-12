from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Define Flask application.
app = Flask(__name__)
cors = CORS(app, origins=["http://localhost:3000"])

# Define tools.

# Serve the index.html file from the templates folder. This file contains our main UI.
@app.route("/")
def hello_world():
    return render_template("index.html")

# Define a route that the end user will interact with.
@app.route("/reply", methods=['POST'])
def reply():

    file = request.files['recording']

    print('File from the POST request is: {}'.format(file))

    response = jsonify(message="POST request returned")

    return response

def audio_to_text(audio):
    return "wow"

def generate_text_reply():
    return "wow2"

def text_to_audio():
    return "wow3"

if __name__ == "__main__":
    app.run(host='localhost', port=5050, debug=True)