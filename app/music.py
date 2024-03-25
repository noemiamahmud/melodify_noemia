import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)


def process_data():
    if(request.method == "POST"):
        input_text = request.form['inputText']  # Access the inputText field from form data
        # Now you can process the input_text as needed
        print('Received input:', input_text)
        # You can return a response if needed
        return 'Data received successfully'