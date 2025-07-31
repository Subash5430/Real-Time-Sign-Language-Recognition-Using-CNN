import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pyttsx3

model = load_model('model/asl_model.h5')
labels = sorted(os.listdir(r'C:\Users\DELL\Desktop\EPICS\Sign Language Detection\dataset\asl_alphabet_train\asl_alphabet_train'))
output_text = ""

engine = pyttsx3.init()


def predict(frame):
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img / 255.0, axis=0)
    prediction = model.predict(img)
    label = labels[np.argmax(prediction)]
    return label


def clear_text():
    global output_text
    output_text = ""
    text_display.delete("1.0", tk.END)


def save_to_file():
    with open("output.txt", "w") as f:
        f.write(output_text)


def speak_text():
    engine.say(output_text)
    engine.runAndWait()


def update_frame():
    global output_text
    ret, frame = cap.read()
    if not ret:
        return
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prediction = predict(cv2image)

    if prediction == "space":
        output_text += " "
    elif prediction != "nothing":
        output_text += prediction

    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    display.imgtk = imgtk
    display.configure(image=imgtk)

    text_display.delete("1.0", tk.END)
    text_display.insert(tk.END, output_text)
    window.after(1000, update_frame)


cap = cv2.VideoCapture(0)
window = tk.Tk()
window.title("Sign Language to Text")

display = tk.Label(window)
display.pack()

text_display = tk.Text(window, height=5, font=("Arial", 20))
text_display.pack()

frame_btn = tk.Frame(window)
frame_btn.pack()

tk.Button(frame_btn, text="Clear All", command=clear_text, bg='lightyellow').pack(side=tk.LEFT, padx=5)
tk.Button(frame_btn, text="Save to a Text File", command=save_to_file, bg='lightgreen').pack(side=tk.LEFT, padx=5)
tk.Button(frame_btn, text="Speak", command=speak_text, bg='lightblue').pack(side=tk.LEFT, padx=5)
tk.Button(frame_btn, text="Quit", command=window.quit, bg='red').pack(side=tk.LEFT, padx=5)

update_frame()
window.mainloop()
cap.release()
cv2.destroyAllWindows()
