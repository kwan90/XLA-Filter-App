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
    text="Advanced Image Processing Application",
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

    elif mode == "popart":

        # smooth image
        smooth = cv2.bilateralFilter(img, 9, 75, 75)

        # reduce colors (posterize)
        div = 64
        pop = smooth // div * div + div // 2

        # edge detect
        gray = cv2.cvtColor(pop, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 80, 180)

        edges = 255 - edges

        # bold black edges
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # combine edge + color
        pop = cv2.bitwise_and(pop, edges_colored)

        # increase saturation
        hsv = cv2.cvtColor(pop, cv2.COLOR_BGR2HSV)

        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)

        pop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # sharpen
        kernel = np.array([
            [-1,-1,-1],
            [-1, 9,-1],
            [-1,-1,-1]
        ])

        processed_img = cv2.filter2D(pop, -1, kernel)

    elif mode == "dragan":

        # smooth nhẹ
        smooth = cv2.bilateralFilter(img, 9, 75, 75)

        # high contrast
        contrast = cv2.convertScaleAbs(
            smooth,
            alpha=1.5,
            beta=-20
        )

        # HPF sharpen
        kernel = np.array([
            [-1,-1,-1],
            [-1, 9,-1],
            [-1,-1,-1]
        ])

        sharp = cv2.filter2D(contrast, -1, kernel)

        # clarity effect
        blur = cv2.GaussianBlur(sharp, (0, 0), 5)

        clarity = cv2.addWeighted(
            sharp,
            1.5,
            blur,
            -0.5,
            0
        )

        # dark cinematic tone
        hsv = cv2.cvtColor(clarity, cv2.COLOR_BGR2HSV)

        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.7, 0, 255)

        processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

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

btn_hpf = Button(
    frame_buttons,
    text="Open Image",
    font=button_font,
    bg="#3498db",
    fg="white",
    width=button_width,
    height=button_height,
    command=open_image
)


btn_hpf = Button(
    frame_buttons,
    text="HPF Sharpen",
    font=button_font,
    bg="#e74c3c",
    fg="white",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter(mode="hpf")
)
btn_hpf.grid(row=1, column=0, padx=8, pady=8)

btn_sobel = Button(
    frame_buttons,
    text="Sobel Edge",
    font=button_font,
    bg="#1abc9c",
    fg="white",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter(mode="sobel")
)
btn_sobel.grid(row=1, column=2, padx=8, pady=8)

btn_canny = Button(
    frame_buttons,
    text="Canny Edge",
    font=button_font,
    bg="#34495e",
    fg="white",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter(mode="canny")
)
btn_canny.grid(row=1, column=3, padx=8, pady=8)

btn_laplacian = Button(
    frame_buttons,
    text="Laplacian",
    font=button_font,
    bg="#16a085",
    fg="white",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter(mode="laplacian")
)
btn_laplacian.grid(row=1, column=4, padx=8, pady=8)

btn_popart = Button(
    frame_buttons,
    text="Pop Art",
    font=button_font,
    bg="#ffcc00",
    fg="black",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter(mode="popart")
)
btn_popart.grid(row=1, column=7, padx=8, pady=8)

btn_sketch = Button(
    frame_buttons,
    text="Sketch",
    font=button_font,
    bg="#c0392b",
    fg="white",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter(mode="sketch")
)
btn_sketch.grid(row=1, column=6, padx=8, pady=8)

btn_dragan = Button(
    frame_buttons,
    text="Dragan Effect",
    font=button_font,
    bg="#555555",
    fg="white",
    width=button_width,
    height=button_height,
    command=lambda: apply_filter(mode="dragan")
)

btn_dragan.grid(row=1, column=8, padx=8, pady=8)

# =========================
# RUN
# =========================

root.mainloop()
