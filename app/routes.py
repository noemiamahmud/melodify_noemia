from flask import Blueprint, render_template, request
import os
import json
from dotenv import load_dotenv


bp = Blueprint("pages", __name__)

@bp.route("/", methods=["GET","POST"])
def process_data():
    input_text = request.form['inputText']  # Access the inputText field from form data
    # Now you can process the input_text as needed
    print('Received input:', input_text)
    # You can return a response if needed
    return 'Data received successfully'


