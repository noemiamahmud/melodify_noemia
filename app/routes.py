from flask import Blueprint, render_template, request, jsonify
from audiocraft.models import MusicGen
from MusicGen import generate_music_tensors, save_audio
import torch
import torchaudio
import os
import json
import logging
import MusicGen
from dotenv import load_dotenv


bp = Blueprint('pages', __name__)

@bp.route("/", methods=["GET"])
def index():
    # Render the index.html template or return any other initial response
    return render_template('pages/index.html')



# Load the pre-trained model
#model = MusicGen.get_pretrained('facebook/musicgen-small')

# Set generation parameters
'''
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=10  # Default duration, can be overridden
)'''

# Function to generate music tensors
'''
def generate_music_tensors(description, duration):
    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )
    return output[0]'''

# Function to save audio samples to a file
'''
def save_audio(samples, save_path, sample_rate=32000):
    torchaudio.save(save_path, samples, sample_rate)'''

@bp.route('/generate_music', methods=['POST'])
def generate_music():
    # Get data from request
    data = request.json
    description = data['description']
    duration = int(data['duration'])

    # Generate music tensors
    music_tensors = generate_music_tensors(description, duration)

    # Save generated music to a file
    save_path = 'generated_audio/'
    save_audio(music_tensors, save_path)

    # Return the path to the generated audio file
    return jsonify({'audio_path': save_path})

