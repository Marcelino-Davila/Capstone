import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Pizza Drone Bomb Finder")
root.geometry("1000x1000")  # width x height

# Add a label
label = tk.Label(root, text="Bomb Finder")
label.pack(pady=10)

# Add an entry box
entry = tk.Entry(root)
entry.pack(pady=5)
# Add a label
label = tk.Label(root, text="Bomb Finder")
label.pack(pady=10)
# Add an entry box
entry = tk.Entry(root)
entry.pack(pady=5)

# Add a button
def on_click():
    user_input = entry.get()
    label.config(text=f"You typed: {user_input}")

button = tk.Button(root, text="Submit", command=on_click)
button.pack(pady=10)

# Run the GUI loop
root.mainloop()