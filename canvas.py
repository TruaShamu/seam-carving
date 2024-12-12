import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageTk
import numpy as np

class SeamCarvingApp:
    def __init__(self, root):
        self.root = root 
        self.root.title("Seam Carving")
        self.root.geometry("600x550")

        self.canvas = tk.Canvas(root, width=400, height=250) # Initialize canvas
        self.canvas.pack(pady=10) # Add padding around the canvas

        self.current_image = None
        self.photo = None
        self.drawing_layer = None

        # Drawing mode
        self.drawing_mode = "draw"

        # Create buttons frame
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=5)

        # Load Image button
        self.load_btn = tk.Button(
            self.button_frame, 
            text="Load Image", 
            command=self.load_image
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Draw button
        self.draw_btn = tk.Button(
            self.button_frame, 
            text="Draw", 
            command=self.set_draw_mode,
            bg="lightgreen"
        )
        self.draw_btn.pack(side=tk.LEFT, padx=5)

        # Eraser button
        self.erase_btn = tk.Button(
            self.button_frame, 
            text="Eraser", 
            command=self.set_erase_mode
        )
        self.erase_btn.pack(side=tk.LEFT, padx=5)

        # Clear All button
        self.clear_btn = tk.Button(
            self.button_frame, 
            text="Clear Drawings", 
            command=self.clear_drawings
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Save Mask button
        self.save_mask_btn = tk.Button(
            self.button_frame, 
            text="Save Mask", 
            command=self.save_mask
        )
        self.save_mask_btn.pack(side=tk.LEFT, padx=5)

        # Brush size slider
        self.size_label = tk.Label(root, text="Brush Size:")
        self.size_label.pack()
        self.size_var = tk.IntVar(value=5)
        self.size_slider = tk.Scale(
            root, 
            from_=1, 
            to=20, 
            orient=tk.HORIZONTAL, 
            variable=self.size_var
        )
        self.size_slider.pack()

        # Bind drawing events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Drawing state
        self.last_x = None
        self.last_y = None


    def save_mask(self):
        if self.drawing_layer:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".npy", 
                filetypes=[("Numpy Files", "*.npy")]
            )
            if save_path:
                mask_array = np.array(self.drawing_layer)[:, :, 3]  # Extract alpha channel
                np.save(save_path, mask_array)
                
    def load_image(self):
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        
        if file_path:
            # Open the image
            image = Image.open(file_path)
            
            # Get image dimensions
            image_width, image_height = image.size
            
            # Resize the canvas to fit the image
            self.canvas.config(width=image_width, height=image_height)

            # Create a transparent drawing layer matching the image size
            self.drawing_layer = Image.new('RGBA', (image_width, image_height), (255, 255, 255, 0))
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(image)
            
            # Clear previous image and draw the new image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # Store the current image
            self.current_image = image


    def set_draw_mode(self):
        self.drawing_mode = "draw"
        # Highlight draw button
        self.draw_btn.config(bg="lightgreen")
        self.erase_btn.config(bg="SystemButtonFace")

    def set_erase_mode(self):
        self.drawing_mode = "erase"
        # Highlight erase button
        self.erase_btn.config(bg="lightcoral")
        self.draw_btn.config(bg="SystemButtonFace")

    def start_draw(self, event):
        # Reset last position when starting to draw
        self.last_x = event.x
        self.last_y = event.y

    def paint(self, event):
        if self.current_image is None:
            return

        # Get current brush size
        size = self.size_var.get()
        
        # Create a drawing context
        draw = ImageDraw.Draw(self.drawing_layer)
        
        if self.drawing_mode == "draw":
            # Draw a line with semi-transparent green
            draw.line(
                [(self.last_x, self.last_y), (event.x, event.y)], 
                fill=(0, 255, 0, 128),  # Green with 50% opacity
                width=size*2
            )
        elif self.drawing_mode == "erase":
            # Create an eraser effect by drawing transparent lines
            draw.line(
                [(self.last_x, self.last_y), (event.x, event.y)], 
                fill=(255, 255, 255, 0),  # Fully transparent
                width=size*2
            )

        # Update the canvas
        self.update_canvas()

        # Update last position
        self.last_x = event.x
        self.last_y = event.y

    def update_canvas(self):
        # Composite the original image with the drawing layer
        if self.current_image and self.drawing_layer:
            composite = Image.alpha_composite(
                self.current_image.convert('RGBA'), 
                self.drawing_layer
            )
            
            # Convert back to PhotoImage
            self.photo = ImageTk.PhotoImage(composite)
            
            # Redraw the canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def clear_drawings(self):
        # Reset drawing layer
        if self.current_image:
            self.drawing_layer = Image.new('RGBA', self.current_image.size, (255, 255, 255, 0))
            self.update_canvas()

# Create the main window
root = tk.Tk()
app = SeamCarvingApp(root)
root.mainloop()