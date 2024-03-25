from flask import Blueprint, render_template, request
import os
import json
from dotenv import load_dotenv


bp = Blueprint('pages', __name__)

@bp.route("/", methods=["GET","POST"])
def createMusic():
    if request.method == 'POST':
        textInput = request.form['textInput']
        print(textInput)
        
    return render_template('index.html')

