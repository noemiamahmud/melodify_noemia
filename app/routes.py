from flask import Blueprint, render_template, request
import os
import json
import logging
import MusicGen
from MusicGen import loadModel, generate_audio
from dotenv import load_dotenv


bp = Blueprint('pages', __name__)

@bp.route("/", methods=["GET", "POST"])
def index():
    userInput = ""
    if request.method == 'POST':
        userInput = request.form.get("userInput")
        #generate_audio(userInput)
        print(userInput)
        logging.info("User input: %s", userInput)

       
    return render_template('pages/index.html', user_input = userInput)

