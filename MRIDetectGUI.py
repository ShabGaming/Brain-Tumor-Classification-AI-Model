import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import webbrowser
import tensorflow as tf  # pip install tensorflow==2.12.0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Define the directories for the dataset
current_dir = os.getcwd()
model_dir = os.path.join(current_dir, 'model.h5')

# Define the target size and batch size
target_size = (1250, 1250)
batch_size = 32

# Load the trained model
model = tf.keras.models.load_model(model_dir)

# The Labels
labels = ['Glioma tumor', 'Meningioma tumor', 'No tumor', 'Pituitary tumor']

# Preprocessing function
def preprocess_input(x):
    x = np.array(x)
    x = x.astype('float32') / 255.
    return x

# Define the classify function
def classify_image():
    # Get the path of the selected image
    image_path = filedialog.askopenfilename(title='Select Image')

    # Load the image and preprocess it
    image = Image.open(image_path).resize(target_size)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Use the trained model to predict the class of the image
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    predicted_label = labels[prediction]

    # Load the image and display it in the GUI
    selected_image = Image.open(image_path).resize((300, 300))
    selected_image = ImageTk.PhotoImage(selected_image)
    selected_image_label.config(image=selected_image)
    selected_image_label.image = selected_image

    # Update the GUI with the predicted label
    result_label.config(text=f'Predicted: {predicted_label}')


# Define the GUI app
root = tk.Tk()
root.title('MRI Brain Tumor Detection')
root.config(bg='#212121')

# Define the icon
icon = ImageTk.PhotoImage(Image.open('icon.png'))
root.iconphoto(True, icon)

# Define the title label
title_label = tk.Label(root, text='MRI Brain Tumor Detection', font=(
    'Helvetica', 24), fg='#fff', bg='#212121')
title_label.pack(pady=10)

# Define the select image button
select_button = tk.Button(root, text='Select Image', font=(
    'Helvetica', 16), fg='#fff', bg='#3f51b5', command=classify_image)
select_button.pack(pady=10)

selected_image_label = tk.Label(root, text='', font=(
    'Helvetica', 16), fg='#fff', bg='#212121')
selected_image_label.pack(pady=10)

# Define the result label
result_label = tk.Label(root, text='', font=(
    'Helvetica', 16), fg='#fff', bg='#212121')
result_label.pack(pady=10)

# Define the footer frame
footer_frame = tk.Frame(root, bg='#212121')
footer_frame.pack(side='bottom', fill='x')

# Define the GitHub label
github_label = tk.Label(footer_frame, text='By github.com/ShabGaming', font=(
    'Helvetica', 12), fg='#fff', bg='#212121')
github_label.pack(side='left', padx=10)

# Make the GitHub URL clickable
def open_github(event):
    import webbrowser
    webbrowser.open_new('https://github.com/ShabGaming')


github_label.bind("<Button-1>", open_github)

# Define the caution label
caution_label = tk.Label(footer_frame, text='Please Do Not Use This For A Diagnosis; Predictions Might Be Inaccurate', font=(
    'Helvetica', 12), fg='red', bg='#212121')
caution_label.pack(side='right', padx=10)

# Define the YouTube icon
youtube_icon = ImageTk.PhotoImage(Image.open('youtube.png').resize((40, 40)))
youtube_button = tk.Button(footer_frame, image=youtube_icon, bd=0, bg='#212121', activebackground='#212121',
                           command=lambda: webbrowser.open_new('https://www.youtube.com/Shabpassiongamer'))
youtube_button.pack(side='right', padx=10)

# Define the Fiverr icon
fiverr_icon = ImageTk.PhotoImage(Image.open('fiverr.png').resize((40, 40)))
fiverr_button = tk.Button(footer_frame, image=fiverr_icon, bd=0, bg='#212121', activebackground='#212121',
                          command=lambda: webbrowser.open_new('https://www.fiverr.com/best_output'))
fiverr_button.pack(side='right', padx=10)

# Define the selected image label
selected_image_label = tk.Label(root, text='', font=(
    'Helvetica', 16), fg='#fff', bg='#212121')
selected_image_label.pack(pady=10)


# Start the GUI app
root.mainloop()