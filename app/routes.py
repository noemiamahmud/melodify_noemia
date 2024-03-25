from flask import Blueprint, render_template, request
import os
import json
import logging
from dotenv import load_dotenv


bp = Blueprint('pages', __name__)
logging.warning("WARNING")

@bp.route("/", methods=["GET", "POST"])
def index():
    userInput = ""
    if request.method == 'POST':
        userInput = request.form.get("userInput")
        print(userInput)
        logging.info("User input: %s", userInput)
        logging.warning("WARNING")

       
    return render_template('pages/test.html', user_input = userInput)

