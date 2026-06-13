import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import tensorflow as tf
import os

# ---------- CHECK MODEL ----------
if not os.path.exists("digit_model.h5"):
    print("❌ Run train_model.py first!")
    exit()

model = tf.keras.models.load_model("digit_model.h5")

# ---------- GUI ----------
root = tk.Tk()
root.title("Digit Recognizer")
root.state('zoomed')

# Canvas
canvas = tk.Canvas(root, bg='black')
canvas.pack(fill="both", expand=True)

root.update()
canvas_width = root.winfo_width()
canvas_height = root.winfo_height()

# Image for drawing
image = Image.new("L", (canvas_width, canvas_height), color=0)
draw = ImageDraw.Draw(image)

# ---------- DRAW ----------
def draw_lines(event):
    x, y = event.x, event.y
    r = 20
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
    draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

canvas.bind("<B1-Motion>", draw_lines)

# ---------- PREPROCESS ----------
def preprocess(img):
    img = np.array(img) / 255.0

    coords = np.argwhere(img > 0.1)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        digit = img[y_min:y_max+1, x_min:x_max+1]

        h, w = digit.shape
        if h > w:
            new_h = 20
            new_w = int(w * (20 / h))
        else:
            new_w = 20
            new_h = int(h * (20 / w))

        digit = Image.fromarray((digit*255).astype(np.uint8))
        digit = digit.resize((new_w, new_h))

        digit = np.array(digit) / 255.0

        new_img = np.zeros((28, 28))
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2

        new_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit
        img = new_img

    return img.reshape(1, 28, 28, 1)

# ---------- PREDICT ----------
def predict_digit():
    img = preprocess(image)
    pred = model.predict(img)
    digit = np.argmax(pred)
    conf = np.max(pred)
    label.config(text=f"Prediction: {digit} ({conf*100:.2f}%)")

# ---------- UPLOAD ----------
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )

    if file_path:
        img = Image.open(file_path).convert('L')

        # Show image on canvas
        display = img.resize((canvas_width, canvas_height))
        display_tk = ImageTk.PhotoImage(display)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=display_tk)
        canvas.image = display_tk

        # Auto invert if needed
        arr = np.array(img)
        if np.mean(arr) > 127:
            img = ImageOps.invert(img)

        processed = preprocess(img)

        pred = model.predict(processed)
        digit = np.argmax(pred)
        conf = np.max(pred)

        label.config(text=f"Uploaded: {digit} ({conf*100:.2f}%)")

# ---------- CLEAR ----------
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=0)

# ---------- CONTROLS ----------
controls = tk.Frame(root, bg="gray")
controls.pack(fill="x")

tk.Button(controls, text="Predict", width=15, command=predict_digit).pack(side="left", padx=10, pady=10)
tk.Button(controls, text="Clear", width=15, command=clear_canvas).pack(side="left", padx=10)
tk.Button(controls, text="Upload Image", width=15, command=upload_image).pack(side="left", padx=10)
tk.Button(controls, text="Exit", width=15, command=root.destroy).pack(side="right", padx=10)

# ---------- OUTPUT ----------
label = tk.Label(root, text="Draw or Upload a Digit", font=("Arial", 20))
label.pack()

root.mainloop()