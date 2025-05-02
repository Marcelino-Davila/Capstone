import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# === Data Model ===
class detectedObject():
    def __init__(self, description, location, RGB, LWIR, RADAR, LIDAR):
        self.description = description
        self.location = location
        self.RGB = RGB
        self.LWIR = LWIR
        self.RADAR = RADAR
        self.LIDAR = LIDAR

# === Viewer GUI (Second Window, XP Style) ===
def open_viewer_gui(objects):
    current_idx = [0]

    viewer = tk.Toplevel()
    viewer.title("Scan Results - Windows XP Style")
    viewer.geometry("1000x700")
    viewer.configure(bg="#d4d0c8")

    title = tk.Label(viewer, text="Detected Object Viewer", font=("Tahoma", 14, "bold"),
                     fg="white", bg="#0a246a", padx=10, pady=5)
    title.pack(fill="x")

    info_frame = tk.Frame(viewer, bg="#d4d0c8")
    info_frame.pack(pady=10)

    desc_label = tk.Label(info_frame, font=("Tahoma", 12), bg="#d4d0c8")
    desc_label.pack()
    loc_label = tk.Label(info_frame, font=("Tahoma", 10), bg="#d4d0c8")
    loc_label.pack()

    image_frame = tk.Frame(viewer, bg="#d4d0c8")
    image_frame.pack(pady=10)

    labels = {}
    images = {}

    for i, modality in enumerate(["RGB", "LWIR", "RADAR", "LIDAR"]):
        tk.Label(image_frame, text=modality, font=("Tahoma", 9, "bold"), bg="#d4d0c8").grid(row=0, column=i)
        labels[modality] = tk.Label(image_frame, bg="#ffffff", relief="sunken", borderwidth=2)
        labels[modality].grid(row=1, column=i, padx=10, pady=5)

    def load_image(path):
        img = Image.open(path).resize((200, 200))
        return ImageTk.PhotoImage(img)

    def update_gui(index):
        obj = objects[index]
        desc_label.config(text=f"Description: {obj.description}")
        loc_label.config(text=f"Location: {obj.location}")
        for modality in ["RGB", "LWIR", "RADAR", "LIDAR"]:
            images[modality] = load_image(getattr(obj, modality))
            labels[modality].config(image=images[modality])
            labels[modality].image = images[modality]

    def go_prev():
        if current_idx[0] > 0:
            current_idx[0] -= 1
            update_gui(current_idx[0])

    def go_next():
        if current_idx[0] < len(objects) - 1:
            current_idx[0] += 1
            update_gui(current_idx[0])

    control_frame = tk.Frame(viewer, bg="#d4d0c8")
    control_frame.pack(pady=10)

    btn_style = {"font": ("Tahoma", 10), "bg": "#f0f0f0", "relief": "raised", "borderwidth": 2}

    tk.Button(control_frame, text="← Previous", command=go_prev, **btn_style).pack(side="left", padx=30)
    tk.Button(control_frame, text="Next →", command=go_next, **btn_style).pack(side="right", padx=30)

    update_gui(current_idx[0])

# === Main Function Placeholder ===
def run_main_function(x1, x2, y1, y2, selected_modalities):
    path1 = r"C:\Users\Madness\Downloads\WallpapersAndOtherRandomImages\Yukio_Mishima,_1955_(cropped).jpg"
    path2 = r"C:\Users\Madness\Downloads\WallpapersAndOtherRandomImages\licensed-image.jpg"
    targets = [
        detectedObject("metal", "0.0,1.1", path1, path1, path1, path1),
        detectedObject("plastic", "1.0,4.1", path2, path2, path2, path2)
    ]
    open_viewer_gui(targets)

# === Initial GUI (XP Style) ===
def launch_gui():
    root = tk.Tk()
    root.title("Pizza Drone Bomb Finder - XP Style")
    root.geometry("400x500")
    root.configure(bg="#d4d0c8")

    tk.Label(root, text="Bomb Finder - Input Search Area", font=("Tahoma", 12, "bold"), bg="#0a246a", fg="white").pack(fill="x", pady=5)

    coord_frame = tk.Frame(root, bg="#d4d0c8")
    coord_frame.pack(pady=10)

    for i, label in enumerate(["x1", "x2", "y1", "y2"]):
        tk.Label(coord_frame, text=label, font=("Tahoma", 10), bg="#d4d0c8").grid(row=i, column=0, sticky="e", pady=2)

    x1_entry = tk.Entry(coord_frame)
    x1_entry.grid(row=0, column=1)
    x2_entry = tk.Entry(coord_frame)
    x2_entry.grid(row=1, column=1)
    y1_entry = tk.Entry(coord_frame)
    y1_entry.grid(row=2, column=1)
    y2_entry = tk.Entry(coord_frame)
    y2_entry.grid(row=3, column=1)

    tk.Label(root, text="Select Modalities", font=("Tahoma", 10, "bold"), bg="#d4d0c8").pack(pady=10)

    modalities = {"Radar": tk.BooleanVar(), "LiDAR": tk.BooleanVar(), "RGB": tk.BooleanVar(), "LWIR": tk.BooleanVar()}
    for text, var in modalities.items():
        tk.Checkbutton(root, text=text, variable=var, font=("Tahoma", 9), bg="#d4d0c8").pack(anchor="w", padx=50)

    def on_run():
        try:
            x1 = float(x1_entry.get())
            x2 = float(x2_entry.get())
            y1 = float(y1_entry.get())
            y2 = float(y2_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Coordinates must be numbers.")
            return

        selected = [name for name, var in modalities.items() if var.get()]
        if not selected:
            messagebox.showwarning("No Modalities", "Please select at least one modality.")
            return

        run_main_function(x1, x2, y1, y2, selected)

    tk.Button(root, text="RUN", command=on_run, font=("Tahoma", 10, "bold"), bg="#f0f0f0", relief="raised", borderwidth=2).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
