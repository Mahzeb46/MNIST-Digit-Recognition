import gradio as gr
import cv2
import numpy as np
from joblib import load

# Load the saved trained model
rf_clf_aug = load('rf_clf_aug.joblib')

# Define prediction function
def predict_digit(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (28, 28))
    image = 255 - image  # Invert colors (like MNIST)
    image = image.flatten().reshape(1, -1)
    prediction = rf_clf_aug.predict(image)
    return int(prediction)

# Build Gradio Interface
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(image_mode='L', width=28, height=28),
    outputs="label",
    live=True
)

iface.launch()
