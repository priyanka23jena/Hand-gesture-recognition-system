import os
import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from tkinter import Label, Button, Toplevel
from PIL import Image, ImageTk, ImageDraw, ImageFont
import copy
from collections import deque, Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Mediapipe setup
mp_hands   = mp.solutions.hands
hands      = mp_hands.Hands(min_detection_confidence=0.8,
                           min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Gesture definitions
gesture_labels = ["Open Hand","Peace","Punch","Waving","Yo"]
gesture_counts = {g:0 for g in gesture_labels}
gesture_history = deque(maxlen=5)

# Build a placeholder dynamically with 3D text
def make_placeholder(width, height):
    img = Image.new("RGB", (width, height), "#222")
    draw = ImageDraw.Draw(img)
    
    # Add 3D text effect (shadow + main text)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()

    text = "Hand Gesture\nRecognition System"
    
    # Shadow effect for 3D look
    shadow_offset = 5
    shadow_color = (50, 50, 50)  # Dark gray shadow
    # Draw shadow
    draw.text((width//2 - 150 + shadow_offset, height//2 - 50 + shadow_offset), 
              text, font=font, fill=shadow_color)
    
    # Draw the main text
    text_color = (255, 255, 255)  # White text
    draw.text((width//2 - 150, height//2 - 50), 
              text, font=font, fill=text_color)

    return img

# Finger‐extended helper
def fingers_extended(L, idxs, m=0.02):
    return all((L.landmark[i].y + m) < L.landmark[i-2].y for i in idxs)

# Classifier
class KeyPointClassifier:
    def __call__(self, L):
        m=0.02
        tt = L.landmark[mp_hands.HandLandmark.THUMB_TIP]
        ip = L.landmark[mp_hands.HandLandmark.THUMB_IP]
        w  = L.landmark[mp_hands.HandLandmark.WRIST]
        tips = [8,12,16,20]
        ext = [fingers_extended(L,[i],m) for i in tips]
        thumb_up = (tt.y + m) < w.y

        if all(ext): return 0
        if ext[0] and ext[1] and not ext[2] and not ext[3]: return 1
        if not any(ext): return 2
        if ext[0] and ext[1] and ext[2] and not ext[3]: return 3
        if (tt.x < ip.x) and ext[3] and not any(ext[:3]): return 4
        
        return -1

classifier = KeyPointClassifier()

# Globals
screen_w, screen_h = pyautogui.size()
cap     = None
running = False
freq_win= None

# Start/Stop
def start_rec():
    global cap, running
    if running: return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return
    running = True
    process_frame()

def stop_rec():
    global cap, running, freq_win
    running = False
    if cap: cap.release()
    cv2.destroyAllWindows()
    show_placeholder()
    if freq_win:
        freq_win.destroy(); freq_win=None

# Frame loop
def process_frame():
    global running
    if not running: return
    ret, frame = cap.read()
    if not ret: return

    frame = cv2.flip(frame,1)
    rgb   = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)
    disp  = frame.copy()

    if res.multi_hand_landmarks:
        for L in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(disp,L,mp_hands.HAND_CONNECTIONS)
            gid = classifier(L)
            gesture_history.append(gid)

    if gesture_history:
        gid,_ = Counter(gesture_history).most_common(1)[0]
        txt = gesture_labels[gid] if gid!=-1 else "No Gesture"
        if gid!=-1:
            gesture_counts[txt]+=1
            update_graph()
    else:
        txt="No Gesture"

    # display
    image = cv2.cvtColor(disp,cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(Image.fromarray(image))
    video_lbl.imgtk=imgtk
    video_lbl.config(image=imgtk)
    gesture_lbl.config(text=f"Gesture: {txt}")

    root.after(20, process_frame)

# Placeholder
def show_placeholder():
    w = video_lbl.winfo_width()  or 640
    h = video_lbl.winfo_height() or 480
    img = make_placeholder(w,h)
    imgtk = ImageTk.PhotoImage(img)
    video_lbl.imgtk=imgtk
    video_lbl.config(image=imgtk)

# Frequency window
def show_freq():
    global freq_win, canvas, ax
    if freq_win: return
    freq_win = Toplevel(root)
    freq_win.title("Gesture Frequency")
    fig,ax = plt.subplots(figsize=(5,3))
    canvas=FigureCanvasTkAgg(fig,master=freq_win)
    canvas.get_tk_widget().pack()
    update_graph()

def update_graph():
    if not freq_win: return
    ax.clear()
    ax.bar(gesture_counts.keys(),gesture_counts.values(),color="orange")
    ax.set_ylim(0,max(gesture_counts.values())+1)
    canvas.draw()

# Build UI
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("900x700")
root.config(bg="#fafafa")

Label(root,text="Hand Gesture Recognition",
      font=("Helvetica",24,"bold"),bg="#fafafa",fg="#333").pack(pady=10)

video_lbl = tk.Label(root,bg="#000")
video_lbl.pack(fill="both",expand=True,padx=20,pady=10)
gesture_lbl = tk.Label(root,text="Gesture: None",
                       font=("Arial",18),bg="#fafafa",fg="#333")
gesture_lbl.pack(pady=5)

btn_frame = tk.Frame(root,bg="#fafafa")
btn_frame.pack(pady=10)
def mkbtn(t,c,col):
    return Button(btn_frame,text=t,command=c,
        font=("Arial",14),bg=col,fg="white",relief="flat",
        width=16,pady=6,activebackground="#666")
mkbtn("Start", start_rec,   "#28a745").grid(row=0,column=0,padx=8)
mkbtn("Stop",  stop_rec,    "#dc3545").grid(row=0,column=1,padx=8)
mkbtn("Show Freq", show_freq,"#007bff").grid(row=0,column=2,padx=8)

root.update()      # let tk initialize sizes
show_placeholder()
root.mainloop()
