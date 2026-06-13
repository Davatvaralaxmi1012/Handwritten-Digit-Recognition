import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

# Create window
root = tk.Tk()
root.title("Handwritten Digit Recognition")

canvas_width = 280
canvas_height = 280

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

image = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image)

def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)

    canvas.create_oval(x1, y1, x2, y2, fill='black')
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

def predict_digit():
    img = image.resize((28, 28))
    img = ImageOps.invert(img)

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    label.config(text=f"Predicted Digit: {digit}")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)
    label.config(text="Draw a digit")

predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.pack()

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

label = tk.Label(root, text="Draw a digit", font=("Arial", 18))
label.pack()

root.mainloop()