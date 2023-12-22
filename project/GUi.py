import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("New_cataract_model.h5")

class CataractDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cataract Detection Web App")

        self.img_width, self.img_height = 224, 224

        # Set a refined color scheme
        self.bg_color = "#EAEAEA"  # Lighter background color
        self.header_color = "#4CAF50"  # Green header color
        self.button_color = "#45a049"  # Darker green button color
        self.label_color = "#333333"  # Dark gray label color

        # GUI components
        self.root.configure(bg=self.bg_color)

        # Header
        self.header_label = tk.Label(
            root,
            text="Cataract Detection Web App",
            bg=self.header_color,
            fg="white",
            font=("Helvetica", 20, "bold"),
            padx=10,
            pady=5,
        )
        self.header_label.pack(fill="x")

        self.choose_file_button = tk.Button(
            root,
            text="Choose Image",
            command=self.predict_image_and_update_gui,
            bg=self.button_color,
            fg="white",
            activebackground=self.button_color,
            padx=10,
            pady=5,
            font=("Helvetica", 14),
        )
        self.choose_file_button.pack(pady=20)

        self.image_label = tk.Label(root, bg=self.bg_color)
        self.image_label.pack()

        self.result_label = tk.Label(root, text="Prediction: ", bg=self.bg_color, fg=self.label_color, font=("Helvetica", 16))
        self.result_label.pack(pady=10)

    def load_and_preprocess_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((self.img_width, self.img_height))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_image_and_update_gui(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img_array = self.load_and_preprocess_image(file_path)
            prediction = model.predict(img_array)
            result_text = f"Prediction: {'Cataract' if prediction > 0.5 else 'Normal'}"
            self.result_label.config(text=result_text)
            self.display_selected_image(file_path)

    def display_selected_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((300, 300))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = CataractDetectionApp(root)
    root.mainloop()
