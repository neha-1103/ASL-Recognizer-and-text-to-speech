import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import time
from collections import deque

# ---------------------------------------------------------
# Load trained CNN model
# ---------------------------------------------------------
MODEL_PATH = "Model/asl_final_model.h5"     # change if different name
model = load_model(MODEL_PATH)

# Class labels (must match folder names from training)
# Example for A–Z (modify if you have different classes)
classes = sorted(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L',
                  'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])

# Smoothing buffer (for stable predictions)
smooth_buffer = deque(maxlen=10)


# ---------------------------------------------------------
# Preprocessing same as your training
# ---------------------------------------------------------
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    norm = resized / 255.0
    reshaped = norm.reshape(1, 64, 64, 1)
    return reshaped


# ---------------------------------------------------------
# GUI Application
# ---------------------------------------------------------
class ASLRealtimeApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("ASL Real-Time Prediction")
        self.window.geometry("900x700")
        self.window.resizable(False, False)

        # Webcam object
        self.cap = cv2.VideoCapture(0)

        # GUI components
        self.video_label = Label(self.window)
        self.video_label.pack(pady=10)

        self.prediction_label = Label(self.window, text="Character: ", font=("Arial", 30))
        self.prediction_label.pack(pady=20)

        self.confidence_label = Label(self.window, text="Confidence: ", font=("Arial", 24))
        self.confidence_label.pack(pady=10)

        # Start loop
        self.update_frame()

        self.window.mainloop()


    # -----------------------------------------------------
    # Real-time frame update loop
    # -----------------------------------------------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)     # mirror view

        # Prediction
        img = preprocess(frame)
        preds = model.predict(img, verbose=0)[0]

        pred_idx = np.argmax(preds)
        pred_char = classes[pred_idx]
        confidence = preds[pred_idx] * 100

        # Add prediction to smoothing queue
        smooth_buffer.append(pred_idx)

        # Compute stable prediction
        stable_idx = max(set(smooth_buffer), key=smooth_buffer.count)
        stable_char = classes[stable_idx]
        stable_conf = preds[stable_idx] * 100

        # Update GUI text
        self.prediction_label.config(text=f"Character: {stable_char}")
        self.confidence_label.config(text=f"Confidence: {stable_conf:.2f}%")

        # Convert frame for Tkinter
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # 10ms delay = smooth 30–50 FPS
        self.window.after(10, self.update_frame)


# ---------------------------------------------------------
# Run the app
# ---------------------------------------------------------
if __name__ == "__main__":
    ASLRealtimeApp()
