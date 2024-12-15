
import os 
import io
import requests
import time
from datetime import datetime
import random

import pandas as pd 
import numpy as np
import cv2 as cv
from PIL import Image
from dotenv import load_dotenv
import joblib

from diffusers import DiffusionPipeline

from model import *
from model import PipelineTester

from flask import Flask, render_template, request, flash, redirect, url_for, get_flashed_messages

load_dotenv()

app = Flask(__name__)

huggingface_token = os.getenv('HUGGINGFACE_API_TOKEN')
secret_key = os.getenv('SECRET_KEY')
api_url = os.getenv('API_URL')

app.config['SECRET_KEY'] = 'abcdef'

API_URL = api_url
headers = {"Authorization": f"Bearer {huggingface_token}"}



@app.route('/')
@app.route('/home')
def home():
    return render_template('dashboard_main.html')


@app.route('/dashboard_main')
def dashboard_main():

    return render_template('dashboard_main.html')


@app.route('/text_image_page', methods=['GET', 'POST'])
def text_image_page():
    input_prompt = None
    image_filename = None
    t_run_time = None

    if request.method == 'POST':
        input_prompt = request.form.get('input_prompt')

        try:
            # Prepare the payload for the API request
            payload = {
                        "inputs": input_prompt,
                        "negative_prompt": "Avoid low resolution, blurry details, oversaturation, poor lighting, and unnatural proportions.",  
                        "num_inference_steps": 35,
                        "width": 480,
                        "height": 640,
                        "guidance_scale": 8
            }
            
            try:
                start_time = datetime.now()
                response = requests.post(API_URL, headers=headers, json=payload)
                end_time = datetime.now()

                # Get the response content (image bytes)
                image_bytes = response.content

                total_run_time = (end_time - start_time).total_seconds()

                # Formatting elapsed time
                if total_run_time < 60:
                    t_run_time = f"{int(total_run_time)} sec"
                else:
                    minutes = int(total_run_time // 60)
                    seconds = int(total_run_time % 60)
                    t_run_time = f"{minutes} minutes {seconds} sec"

            except Exception as e:
                print(f"HuggingFace API request: {e}")

            # save the generated image bytes in png format
            try:
                # Generating a random number for the filename
                ran_num = random.randint(100, 99999)
                random_filename = f"image_{ran_num}.png"
                
                with open('./static/generated_images/'+random_filename, 'wb') as f:
                    f.write(image_bytes)
            except Exception as e:
                print(f"Generated image saving issue: {e}")

        except Exception as e:
            print(e)
        
        # Define the directory where images are saved
        images_dir = "./static/generated_images"
        
        try:
            if os.path.exists(images_dir):
                
                # List and sort image files
                image_files = sorted(
                    (os.path.join(images_dir, f) for f in os.listdir(images_dir)),
                    key=os.path.getmtime,
                    reverse=True
                )

                # If there are files, get the most recent one
                if image_files:
                    image_filename = image_files[0]
                    image_filename = image_filename.replace("\\", "/")
                    image_filename = image_filename.split('/')[-1]
                    # print(image_filename)

        except Exception as e:
            print("Error accessing image directory:", e)
            return render_template('txt_img.html', message="Image Directory Path Issue!")
    
    # print("Input prompt:", input_prompt)
    return render_template('txt_img.html', input_prompt=input_prompt, image_filename=image_filename, t_run_time=t_run_time)



@app.route('/dashboard', methods=['GET', 'POST'])
def credit_fraud():
    if request.method == 'POST':

        flash_fill = request.form.get('flash_fill')

        if flash_fill:

            try:
                values = [x for x in flash_fill.split(',')]

                (category, Amt, gender, lat, log, City_Pop, Job, Year, Hour, Day, Month, Age) = values
            except Exception as e:
                print(e)
        else:
            category = request.form['category']
            gender = request.form['gender']
            Age = int(request.form['Age'])
            Job = request.form['job']
            Amt = float(request.form['amt'])
            Year = int(request.form['Year'])
            lat = float(request.form['lat'])
            log = float(request.form['long'])
            City_Pop = float(request.form['city_pop'])
            Month = int(request.form['Month'])
            Day = int(request.form['Day'])
            Hour = int(request.form['Hour'])
            
        # Creating a DataFrame with the input data
        input_data = pd.DataFrame({
            'category': [category],
            'amt': [Amt],
            'gender': [gender],
            'lat': [lat],
            'long': [log],
            'city_pop': [City_Pop],
            'job': [Job],
            'Year': [Year],
            'Hour': [Hour],
            'Day': [Day],
            'Month': [Month],
            'Age': [Age]
        })
            

        try:
            # Initialize the PipelineTester with the trained pipeline and input data
            model_pipe = PipelineTester('binary_pipeline2.joblib', input_data)

            # Predict the class label for the input data
            prediction_prob = model_pipe.predict()

            # Set a threshold for classification
            threshold = 0.5

            # Apply a threshold to classify the sample
            if prediction_prob > threshold:
                msg = "This is a Fraudulent Transaction"
                flash(msg, category='error')
            else:
                msg="Legitimate Transaction"
                flash(msg, category='success')

        except Exception as e:
            flash(f"Prediction Failed: {str(e)}", category='error')

    return render_template('dashboard.html')



@app.route('/about')
def about():
    
    return render_template('about.html')



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
