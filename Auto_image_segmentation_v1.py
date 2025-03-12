import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Image Segmentation")

        # Initial image settings
        self.image_path = None
        self.image = None
        self.segmented_image = None
        self.selected_color = None
        self.history = []  # Stack to keep track of previous segmented images
        self.remove_mask_mode = False  # Flag to check if remove mask mode is active

        # Set up the canvas to display the image
        self.canvas = tk.Canvas(root, width=1200, height=700)
        self.canvas.pack()

        # Buttons
        self.open_button = tk.Button(root, text="Open Image", command=self.open_image)
        self.open_button.pack()

        self.next_button = tk.Button(root, text="Next", command=self.reset_white_regions)
        self.next_button.pack()

        self.undo_button = tk.Button(root, text="Undo", command=self.undo)
        self.undo_button.pack()

        self.remove_mask_button = tk.Button(root, text="Remove Green Mask", command=self.toggle_remove_mask_mode)
        self.remove_mask_button.pack()

        self.save_mask_button = tk.Button(root, text="Save Green Mask", command=self.save_green_mask)
        self.save_mask_button.pack()

        self.canvas.bind("<Button-1>", self.on_click)

    def open_image(self):
        """ Open image from file dialog and display it """
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            self.image = cv2.imread(self.image_path)
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.segmented_image = np.copy(self.image_rgb)  # Keep original for resetting white regions
            self.history = [self.segmented_image.copy()]  # Save the initial state for undo
            self.display_image(self.image_rgb)

    def on_click(self, event):
        """ Handle click on the image and select a point for auto segmentation or remove mask """
        if self.image_rgb is None:
            print("Image not loaded!")
            return

        x, y = event.x, event.y
        print(f"Selected point: ({x}, {y})")

        if self.remove_mask_mode:
            self.remove_mask(x, y)
        else:
            self.selected_color = self.image_rgb[y, x]
            print(f"Color of selected point: {self.selected_color}")
            self.segment_image(x, y)

    def segment_image(self, x, y):
        """ Perform color-based segmentation automatically in a small region around the click """
        if self.image is None or self.selected_color is None:
            return

        self.history.append(self.segmented_image.copy())

        region_size = 50
        height, width, channels = self.image_rgb.shape

        x_start = max(x - region_size // 2, 0)
        x_end = min(x + region_size // 2, width)
        y_start = max(y - region_size // 2, 0)
        y_end = min(y + region_size // 2, height)

        region = self.image_rgb[y_start:y_end, x_start:x_end]
        region_pixels = region.reshape((-1, 3))

        distance = np.linalg.norm(region_pixels - self.selected_color, axis=1)
        threshold = 300  
        labels = distance < threshold  

        segmented_region = np.copy(region_pixels)
        segmented_region[labels] = [0, 255, 0]  # Green for similar pixels
        segmented_region[~labels] = [255, 255, 255]  # White for background pixels

        segmented_region = segmented_region.reshape((y_end - y_start, x_end - x_start, 3))
        self.segmented_image[y_start:y_end, x_start:x_end] = segmented_region

        self.display_image(self.segmented_image)

    def reset_white_regions(self):
        """ Reset the white regions and keep green regions from previous segmentation """
        if self.segmented_image is None:
            return

        mask = np.all(self.segmented_image == [255, 255, 255], axis=-1)
        self.segmented_image[mask] = self.image_rgb[mask]

        self.display_image(self.segmented_image)

    def undo(self):
        """ Undo the previous segmentation """
        if len(self.history) > 1:
            self.history.pop()
            self.segmented_image = self.history[-1]
            self.display_image(self.segmented_image)

    def toggle_remove_mask_mode(self):
        """ Toggle the mode for removing green mask """
        self.remove_mask_mode = not self.remove_mask_mode
        if self.remove_mask_mode:
            print("Remove Mask mode is ON.")
        else:
            print("Remove Mask mode is OFF.")

    def remove_mask(self, x, y):
        """ Remove the green mask from the image by resetting the region under the cursor """
        region_size = 20
        height, width, channels = self.image_rgb.shape

        x_start = max(x - region_size // 2, 0)
        x_end = min(x + region_size // 2, width)
        y_start = max(y - region_size // 2, 0)
        y_end = min(y + region_size // 2, height)

        self.segmented_image[y_start:y_end, x_start:x_end] = self.image_rgb[y_start:y_end, x_start:x_end]

        self.display_image(self.segmented_image)

    def save_green_mask(self):
        """ Convert the green mask into a binary image and save as PNG """
        if self.segmented_image is None:
            print("No segmented image to save!")
            return

        # Create a binary mask
        green_pixels = np.all(self.segmented_image == [0, 255, 0], axis=-1)
        binary_mask = np.zeros_like(self.segmented_image, dtype=np.uint8)
        binary_mask[green_pixels] = [255, 255, 255]  # White for segmented areas
        binary_mask[~green_pixels] = [0, 0, 0]  # Black for the background

        # Convert to grayscale
        binary_mask_gray = cv2.cvtColor(binary_mask, cv2.COLOR_RGB2GRAY)

        # Ask user where to save
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, binary_mask_gray)
            print(f"Binary mask saved as: {file_path}")

    def display_image(self, img):
        """ Convert image to a format Tkinter can display and show it on canvas at original size """
        img = Image.fromarray(img)
        
        # Get original image dimensions
        width, height = img.size

        # Update canvas size to match the image
        self.canvas.config(width=width, height=height)

        img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Keep reference to avoid garbage collection

# Create the Tkinter window and app instance
root = tk.Tk()
app = ImageSegmentationApp(root)
root.mainloop()
