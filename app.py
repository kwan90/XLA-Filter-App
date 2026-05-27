import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# =========================
# WINDOW
# =========================

root = Tk()
root.title("Filter App")
root.geometry("1400x850")
root.configure(bg="#1e1e1e")

# =========================
# GLOBALS
# =========================

original_img = None
processed_img = None

# =========================
# TITLE
# =========================

title = Label(
    root,
    text="HPF Filter App",
    font=("Arial", 24, "bold"),
    bg="#1e1e1e",
    fg="white"
)

title.pack(pady=10)

# =========================
# IMAGE AREA
# =========================

frame_images = Frame(root, bg="#1e1e1e")
frame_images.pack(pady=10)

# LEFT PANEL
frame_left = Frame(frame_images, bg="#2b2b2b")
frame_left.pack(side=LEFT, padx=20)

label_original = Label(
    frame_left,
    text="Original Image",
    font=("Arial", 16, "bold"),
    bg="#2b2b2b",
    fg="white"
)
label_original.pack(pady=10)

panel_original = Label(frame_left, bg="black")
panel_original.pack()

# RIGHT PANEL
frame_right = Frame(frame_images, bg="#2b2b2b")
frame_right.pack(side=RIGHT, padx=20)

label_processed = Label(
    frame_right,
    text="Processed Image",
    font=("Arial", 16, "bold"),
    bg="#2b2b2b",
    fg="white"
)
label_processed.pack(pady=10)

panel_processed = Label(frame_right, bg="black")
panel_processed.pack()

# =========================
# DISPLAY IMAGE
# =========================


def display_image(img, panel):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_pil = Image.fromarray(img_rgb)

    img_pil.thumbnail((600, 500))

    imgtk = ImageTk.PhotoImage(img_pil)

    panel.configure(image=imgtk)
    panel.image = imgtk


# =========================
# OPEN IMAGE
# =========================


def open_image():

    global original_img, processed_img

    path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.png *.jpeg")
        ]
    )

    if not path:
        return

    original_img = cv2.imread(path)

    processed_img = original_img.copy()

    display_image(original_img, panel_original)
    display_image(processed_img, panel_processed)


# =========================
# RESET IMAGE
# =========================


def reset_image():

    global processed_img

    if original_img is None:
        return

    processed_img = original_img.copy()

    display_image(processed_img, panel_processed)


# =========================
# SAVE IMAGE
# =========================


def save_image():

    global processed_img

    if processed_img is None:
        messagebox.showerror(
            "Error",
            "No processed image"
        )
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[
            ("JPG File", "*.jpg"),
            ("PNG File", "*.png")
        ]
    )

    if not path:
        return

    cv2.imwrite(path, processed_img)

    messagebox.showinfo(
        "Saved",
        "Image saved successfully"
    )



# =========================
# FILTER FUNCTIONS
# =========================

def apply_filter(mode="hpf"):

    global processed_img

    if original_img is None:
        messagebox.showerror(
            "Error",
            "Please open an image first"
        )
        return

    img = original_img.copy()

    # =========================
    # HPF SHARPEN
    # =========================

    if mode == "hpf":

        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])

        processed_img = cv2.filter2D(img, -1, kernel)

    # =========================
    # SOBEL EDGE
    # =========================

    elif mode == "sobel":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        sobel = cv2.magnitude(sobelx, sobely)

        sobel = np.uint8(np.clip(sobel, 0, 255))

        processed_img = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    # =========================
    # CANNY EDGE
    # =========================

    elif mode == "canny":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 200)

        processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # =========================
    # LAPLACIAN
    # =========================

    elif mode == "laplacian":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        lap = cv2.Laplacian(gray, cv2.CV_64F)

        lap = np.uint8(np.absolute(lap))

        processed_img = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

    # =========================
    # ADVANCED SKETCH
    # =========================

    elif mode == "sketch":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        inverted = 255 - gray

        blur = cv2.GaussianBlur(inverted, (25, 25), 0)

        inverted_blur = 255 - blur

        sketch = cv2.divide(gray, inverted_blur, scale=256.0)

        # sharpen sketch
        kernel = np.array([
            [-1,-1,-1],
            [-1, 9,-1],
            [-1,-1,-1]
        ])

        sketch = cv2.filter2D(sketch, -1, kernel)

        processed_img = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    display_image(processed_img, panel_processed)

# =========================
# BUTTON STYLES
# =========================

button_font = ("Arial", 12, "bold")
button_width = 16
button_height = 2

# =========================
# BUTTONS FRAME
# =========================

frame_buttons = Frame(root, bg="#1e1e1e")
frame_buttons.pack(pady=20)

# =========================
# ROW 1 - MAIN FUNCTIONS
# =========================

# CENTER ROW 1
frame_main_buttons = Frame(frame_buttons, bg="#1e1e1e")
frame_main_buttons.grid(row=0, column=0, columnspan=7, pady=5)

btn_open = Button(
    frame_main_buttons,
    text="Open Image",
    font=button_font,
    bg="#3498db",
    fg="white",
    width=button_width,
    height=button_height,
    command=open_image
)
btn_open.grid(row=0, column=0, padx=12, pady=8)

btn_save = Button(
    frame_main_buttons,
    text="Save Image",
    font=button_font,
    bg="#2ecc71",
    fg="white",
    width=button_width,
    height=button_height,
    command=save_image
)
btn_save.grid(row=0, column=1, padx=12, pady=8)

btn_reset = Button(
    frame_main_buttons,
    text="Reset Image",
    font=button_font,
    bg="#f39c12",
    fg="white",
    width=button_width,
    height=button_height,
    command=reset_image
)
btn_reset.grid(row=0, column=2, padx=12, pady=8)

# =========================
# ROW 2 - FILTERS
# =========================

frame_filters = Frame(frame_buttons, bg="#1e1e1e")
frame_filters.grid(row=1, column=0, columnspan=7, pady=10)

btn_hpf = Button(
    frame_filters,
    text="HPF Sharpen",
    font=button_font,
    bg="#e74c3c",
    fg="white",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter("hpf")
)
btn_hpf.pack(side=LEFT, padx=8)

btn_sobel = Button(
    frame_filters,
    text="Sobel Edge",
    font=button_font,
    bg="#1abc9c",
    fg="white",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter("sobel")
)
btn_sobel.pack(side=LEFT, padx=8)

btn_canny = Button(
    frame_filters,
    text="Canny Edge",
    font=button_font,
    bg="#34495e",
    fg="white",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter("canny")
)
btn_canny.pack(side=LEFT, padx=8)

btn_laplacian = Button(
    frame_filters,
    text="Laplacian",
    font=button_font,
    bg="#16a085",
    fg="white",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter("laplacian")
)
btn_laplacian.pack(side=LEFT, padx=8)

btn_sketch = Button(
    frame_filters,
    text="Sketch",
    font=button_font,
    bg="#c0392b",
    fg="white",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter("sketch")
)
btn_sketch.pack(side=LEFT, padx=8)

# =========================
# RUN
# =========================

root.mainloop()
