import tkinter as tk
import numpy as np
import spectral
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from tkinter import filedialog, ttk, Checkbutton, simpledialog, messagebox

root = tk.Tk()
root.title('ENVI Max')
root.geometry('1200x800')

# Create the main frame
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)
frame.grid_rowconfigure(0, weight=0)
frame.grid_rowconfigure(1, weight=1)
frame.grid_columnconfigure(0, weight=1)

# Menu
my_menu = tk.Menu(root)
root.config(menu=my_menu)
file_menu = tk.Menu(my_menu)
edit_menu = tk.Menu(my_menu)
view_menu = tk.Menu(my_menu)

# Right menu label
right_menu_label = tk.Label(frame)
hdr_text_box = tk.Text(right_menu_label)
hdr_Info_Label_scrollbar = ttk.Scrollbar(right_menu_label)
red_combo_box = ttk.Combobox(right_menu_label, state='readonly')
green_combo_box = ttk.Combobox(right_menu_label, state='readonly')
blue_combo_box = ttk.Combobox(right_menu_label, state='readonly')

# Image label for the RGB image
fig, ax = plt.subplots()
ax.axis('off')  # Hide the axes for a cleaner image display
image_label = FigureCanvasTkAgg(fig, master=frame)
plt_toolbar = NavigationToolbar2Tk(image_label, frame)
plt_toolbar.update()
plt_toolbar.grid(row=2, column=0)
image_label.draw()
image_label.get_tk_widget().grid(row=1, column=0, sticky="nswe")

# Spectral Window
spectral_window = tk.Toplevel(root)
spectral_window.title("Spectral Plot")

# Create a matplotlib figure for the spectral plot
spectral_figure = plt.Figure(figsize=(6, 4), dpi=100)
spectral_ax = spectral_figure.add_subplot(111)
spectral_ax.set_title('Spectral Data')
spectral_ax.set_xlabel('Band')
spectral_ax.set_ylabel('Intensity')

# Create a canvas to embed the matplotlib figure in Tkinter
spectral_canvas = FigureCanvasTkAgg(spectral_figure, master=spectral_window)
spectral_canvas.draw()
spectral_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Rectangle size X and Y plus input button
rectangle_size_y = tk.Entry(right_menu_label)
rectangle_size_x = tk.Entry(right_menu_label)

# Load info dark and wight
dark_text = tk.Label(right_menu_label, text="No raw file", bg='lightgray')
white_text = tk.Label(right_menu_label, text="No raw file", bg='lightgray')
dw_correction = tk.Label(right_menu_label, text="Not correction", bg='lightgray')

# Main variables
spec_img = None
rgb_image = None
image_info = {}
rect_position = np.array([(0, 0, 0, 0)], dtype=[('x1', 'i4'), ('y1', 'i4'), ('x2', 'i4'), ('y2', 'i4')])
checker_single_block = tk.IntVar()
checker_add_new_spec = tk.IntVar()
spectral_data = None
rectangle_selector = None
dark_raw = None
white_raw = None


def open_file():
    global spec_img
    global image_info
    global dark_text
    global white_text
    global dark_raw
    global white_raw
    global dw_correction
    """Handles file opening logic."""
    file_path = filedialog.askopenfilename(title='Open File',
                                           filetypes=(("Spectra File", "*.hdr"), ("All files", "*.*")))

    if file_path.endswith('.hdr'):
        # Load the image and metadata
        spec_img = spectral.open_image(file_path).load()

        # Extract the metadata dictionary
        metadata = spec_img.metadata

        default_bands = metadata.get("default bands", [])
        if default_bands:
            default_bands = [int(band) for band in default_bands]

        # Extract relevant metadata fields
        image_info = {
            "samples": int(metadata.get("samples", 0)),
            "bands": int(metadata.get("bands", 0)),
            "lines": int(metadata.get("lines", 0)),
            "data type": int(metadata.get("data type", "Unknown")),
            "default bands": default_bands,
            "wavelengths": metadata.get("wavelength", []),
        }

        display_hdr_info(image_info)
        display_image()

        dark_text.config(text="No raw file", bg='lightgray')
        white_text.config(text="No raw file", bg='lightgray')
        dw_correction.config(text="No correction", bg='lightgray')
        dark_raw = None
        white_raw = None

    # Clear the old plot
    spectral_ax.clear()


def open_spectral_window():
    global spectral_window
    global spectral_figure
    global spectral_ax
    global spectral_canvas
    if spectral_window is None or not spectral_window.winfo_exists():
        spectral_window = tk.Toplevel(root)
        spectral_window.title("Spectral Plot")

        # Create a matplotlib figure for the spectral plot
        spectral_figure = plt.Figure(figsize=(6, 4), dpi=100)
        spectral_ax = spectral_figure.add_subplot(111)
        spectral_ax.set_title('Spectral Data')
        spectral_ax.set_xlabel('Band')
        spectral_ax.set_ylabel('Intensity')

        # Create a canvas to embed the matplotlib figure in Tkinter
        spectral_canvas = FigureCanvasTkAgg(spectral_figure, master=spectral_window)
        spectral_canvas.draw()
        spectral_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bind the window's focus event to ensure it stays below the main window
        spectral_window.bind("<FocusIn>", lambda e: root.lower(spectral_window))


def display_hdr_info(hdr_info):
    wavelengths = "\n".join([f"{i + 1}: {w}" for i, w in enumerate(hdr_info['wavelengths'])])
    info_text = (
        f"Samples: {hdr_info['samples']}\n"
        f"Bands: {hdr_info['bands']}\n"
        f"Lines: {hdr_info['lines']}\n"
        f"Default Bands: {hdr_info['default bands']}\n"
        f"Data type: {hdr_info['data type']}\n"
        f"Wavelengths:\n{wavelengths}\n"
    )
    hdr_text_box.config(state=tk.NORMAL)
    hdr_text_box.delete(1.0, tk.END)
    hdr_text_box.insert(tk.END, info_text)
    hdr_text_box.config(state=tk.DISABLED)
    red_combo_box.config(values=hdr_info['wavelengths'])
    green_combo_box.config(values=hdr_info['wavelengths'])
    blue_combo_box.config(values=hdr_info['wavelengths'])
    if hdr_info['default bands']:
        red_combo_box.current(hdr_info['default bands'][0] - 1)
        green_combo_box.current(hdr_info['default bands'][1] - 1)
        blue_combo_box.current(hdr_info['default bands'][2] - 1)
    else:
        wavelengths = hdr_info.get("wavelengths", [])
        if isinstance(wavelengths, list) and len(wavelengths) == 1 and isinstance(wavelengths[0], str):
            # Split single string by newlines if wavelengths is a single-element list with newline-separated string
            wavelengths = wavelengths[0].splitlines()

        hdr_info["wavelengths"] = [int(float(wave)) for wave in wavelengths]  # Convert to integers
        target_value = 430
        closest_index = min(enumerate(hdr_info["wavelengths"]), key=lambda x: abs(x[1] - (target_value + 200)))[0]
        red_combo_box.current(closest_index)
        closest_index = min(enumerate(hdr_info["wavelengths"]), key=lambda x: abs(x[1] - (target_value + 100)))[0]
        green_combo_box.current(closest_index)
        closest_index = min(enumerate(hdr_info["wavelengths"]), key=lambda x: abs(x[1] - target_value))[0]
        blue_combo_box.current(closest_index)


def display_image():
    global rgb_image
    bands = [red_combo_box.current(), green_combo_box.current(), blue_combo_box.current()]
    rgb_image = np.zeros((image_info["lines"], image_info["samples"], 3), dtype=np.float32)
    for i, band_index in enumerate(bands):
        band_data = np.squeeze(spec_img[:, :, band_index])  # Extract and squeeze band data to ensure 2D shape
        max_value = np.amax(band_data)  # Get the maximum value for normalization

        # Avoid division by zero
        if max_value > 0:
            rgb_image[:, :, i] = band_data / max_value

        # Scale to 0-255 range and convert to uint8 for display
    rgb_image = (rgb_image * 255).astype(np.uint8)
    ax.imshow(rgb_image)  # Display the RGB image on the axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    image_label.draw()


def on_select(eclick, erelease):
    # These should be the coordinates of the drawn rectangle
    rect_position['x1'][0], rect_position['y1'][0] = eclick.xdata, eclick.ydata
    rect_position['x2'][0], rect_position['y2'][0] = erelease.xdata, erelease.ydata

    x_size, y_size = int(rect_position['x2'][0] - rect_position['x1'][0]), int(
        rect_position['y2'][0] - rect_position['y1'][0])
    rectangle_size_x.delete(0, tk.END)
    rectangle_size_y.delete(0, tk.END)
    rectangle_size_x.insert(0, str(x_size))
    rectangle_size_y.insert(0, str(y_size))


def set_rectangle_size():
    if rect_position['x1'][0] != 0 or rect_position['y1'][0] != 0:
        x_size = int(rectangle_size_x.get())
        y_size = int(rectangle_size_y.get())
        rect_position['x2'][0] = rect_position['x1'][0] + x_size
        rect_position['y2'][0] = rect_position['y1'][0] + y_size
        rectangle_selector.extents = (rect_position['x1'][0],
                                      rect_position['x2'][0],
                                      rect_position['y1'][0],
                                      rect_position['y2'][0])


def on_click(event):
    if event.inaxes:  # Ensure the click is within an axis
        x, y = int(event.xdata), int(event.ydata)

    if checker_single_block.get() == 1 and spectral_window.winfo_exists():
        plot_single_spectral(x, y)


def plot_single_spectral(x, y):
    global spectral_data
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    wavelengths = list(map(float, image_info['wavelengths']))

    # Clear the old plot
    spectral_ax.clear()

    # Extract the spectral data for the clicked pixel
    new_spectral = spec_img[y, x, :]  # Should be 1D or at most 2D if required to add axis

    # Ensure new_spectral has consistent dimensions
    if new_spectral.ndim == 1:
        new_spectral = new_spectral[np.newaxis, :]  # Make it 2D (1, num_wavelengths)

    if checker_add_new_spec.get() == 1 and spectral_data is not None:
        # Ensure spectral_data has consistent dimensions
        if spectral_data.ndim == 1:
            spectral_data = spectral_data[np.newaxis, :]  # Make it 2D if needed

        # Concatenate along the first axis
        spectral_data = np.vstack((spectral_data, new_spectral))
    else:
        # Start a new spectral data set
        spectral_data = new_spectral  # Set as the new data directly

    # Plot the spectral data
    for i in range(spectral_data.shape[0]):  # Loop over each spectrum in spectral_data
        color = colors[i % len(colors)] if checker_add_new_spec.get() == 1 else 'm'
        spectral_ax.plot(wavelengths, spectral_data[i].flatten(), color=color)  # Flatten ensures 1D array

    # Set the x-axis ticks every 8 steps or as per wavelengths length
    step = max(1, len(wavelengths) // 8)
    spectral_ax.set_xticks(wavelengths[::step])

    # Draw vertical lines for selected wavelengths
    spectral_ax.axvline(wavelengths[red_combo_box.current()], color='r', linestyle='--', linewidth=1.5)
    spectral_ax.axvline(wavelengths[green_combo_box.current()], color='g', linestyle='--', linewidth=1.5)
    spectral_ax.axvline(wavelengths[blue_combo_box.current()], color='b', linestyle='--', linewidth=1.5)

    # Set plot labels and title
    spectral_ax.set_title("Spectral Data")
    spectral_ax.set_xlabel("Wavelength")
    spectral_ax.set_ylabel("Intensity")

    # Redraw the canvas to show updated plot
    spectral_canvas.draw()
    root.lower(spectral_window)


def plot_rectangle_spectral():
    global spectral_data
    checker_single_block.set(0)
    checker_add_new_spec.set(0)

    # Clear the old plot
    spectral_ax.clear()

    # Extract data from the specified rectangle
    spectral_data = spec_img[
                    rect_position['y1'][0]:rect_position['y2'][0],
                    rect_position['x1'][0]:rect_position['x2'][0],
                    :
                    ]

    wavelengths = list(map(float, image_info['wavelengths']))

    # Verify if spectral_data is not empty
    if spectral_data.size == 0:
        print("No data in the selected region.")
        return  # Exit the function early if there's no data

    # Handle spectral_data dimensions
    if spectral_data.ndim == 3:  # Region selection
        for i in range(spectral_data.shape[0]):
            for j in range(spectral_data.shape[1]):
                plot = spectral_data[i, j, :].flatten()  # Ensure plot data is 1D
                spectral_ax.plot(wavelengths, plot, 'm')

    elif spectral_data.ndim == 2:  # Single row or column selection
        plot = spectral_data[0, :].flatten()  # Use the first row
        spectral_ax.plot(wavelengths, plot, 'm')

    elif spectral_data.ndim == 1:  # Single pixel or edge case
        spectral_ax.plot(wavelengths, spectral_data.flatten(), 'm')

    # Set the x-axis ticks every 8 steps or as per wavelengths length
    step = max(1, len(wavelengths) // 8)
    spectral_ax.set_xticks(wavelengths[::step])

    # Draw vertical lines for selected wavelengths
    spectral_ax.axvline(wavelengths[red_combo_box.current()], color='r', linestyle='--', linewidth=1.5)
    spectral_ax.axvline(wavelengths[green_combo_box.current()], color='g', linestyle='--', linewidth=1.5)
    spectral_ax.axvline(wavelengths[blue_combo_box.current()], color='b', linestyle='--', linewidth=1.5)

    # Set plot labels and title
    spectral_ax.set_title("Spectral Data")
    spectral_ax.set_xlabel("Wavelength")
    spectral_ax.set_ylabel("Intensity")

    # Redraw the canvas to show updated plot
    spectral_canvas.draw()

    # Reset spectral_data
    spectral_data = None
    root.lower(spectral_window)


def clear_spectral_plot():
    global spectral_data
    # Clear the old plot
    spectral_ax.clear()
    spectral_canvas.draw()
    spectral_data = None


def save_image_as_raw():
    """Save the cropped image as RAW and HDR."""
    if spec_img is None:
        messagebox.showerror("Error", "No image to save!")
        return

    cropped_array = spec_img

    # Ask for save location
    save_path = filedialog.asksaveasfilename(defaultextension=".raw",
                                             filetypes=[("RAW files", "*.raw"), ("All files", "*.*")])
    if save_path:
        # Save the RAW file
        cropped_array.astype(np.float32).tofile(save_path)

        # Save HDR file
        hdr_path = save_path.replace(".raw", ".hdr")
        hdr_content = generate_hdr_metadata(cropped_array)

        with open(hdr_path, "w") as hdr_file:
            hdr_file.write(hdr_content)

        messagebox.showinfo("Image Saved",
                            f"RAW image and HDR file have been saved.\nRAW file: {save_path}\nHDR file: {hdr_path}")


def save_cropped_image_as_raw():
    """Save the cropped image as RAW and HDR."""
    if rect_position['x2'][0] > rect_position['x1'][0] or rect_position['y2'] > rect_position['y1']:
        # Get the extents from the rectangle
        x_min, x_max, y_min, y_max = rectangle_selector.extents

        # Convert extents to a NumPy array
        cropped_array = spec_img[int(y_min):int(y_max), int(x_min):int(x_max), :]

        # Ask the user for a filename to save as RAW
        save_path = filedialog.asksaveasfilename(defaultextension=".raw",
                                                 filetypes=[("RAW files", "*.raw"), ("All files", "*.*")])
        if save_path:
            # Save the RAW file
            cropped_array.astype(np.float32).tofile(save_path)  # Save as binary raw data

            # Create the HDR file path
            hdr_path = save_path.replace(".raw", ".hdr")

            # Generate the HDR metadata
            hdr_content = generate_hdr_metadata(cropped_array)

            # Write the HDR metadata to a file
            with open(hdr_path, "w") as hdr_file:
                hdr_file.write(hdr_content)

            tk.messagebox.showinfo("Image Saved",
                                   f"RAW image and HDR file have been saved.\nRAW file: {save_path}\nHDR file: {hdr_path}")


def generate_hdr_metadata(array):
    """Generate HDR metadata for the cropped image."""
    height, width = array.shape[:2]
    bands = 1 if len(array.shape) == 2 else array.shape[2]
    data_type = 4

    # Join wavelengths into a comma-separated string
    wavelengths = ",\n".join(str(wave) for wave in image_info['wavelengths'])

    hdr_content = (
        f"ENVI\n"
        f"description = {{Cropped Image from Application}}\n"
        f"samples = {width}\n"
        f"lines   = {height}\n"
        f"bands   = {bands}\n"
        f"header offset = 0\n"
        f"file type = ENVI Standard\n"
        f"data type = {data_type}\n"
        f"interleave = bip\n"
        f"byte order = 0\n"
        f"wavelength = {{\n"
        f"{wavelengths}\n"
        f"}}"
    )
    return hdr_content


def open_dark_raw():
    global dark_raw

    file_path = filedialog.askopenfilename(title='Open File',
                                           filetypes=(("Spectra File", "*.hdr"), ("All files", "*.*")))
    if file_path.endswith('.hdr'):
        # Load the image and metadata
        dark_raw = spectral.open_image(file_path).load()  # Convert to NumPy array
        dark_text.config(text=f"Uploaded: {dark_raw.shape}", fg='green')
        print(dark_raw.shape)


def open_white_raw():
    global white_raw

    file_path = filedialog.askopenfilename(title='Open File',
                                           filetypes=(("Spectra File", "*.hdr"), ("All files", "*.*")))
    if file_path.endswith('.hdr'):
        # Load the image and metadata
        white_raw = spectral.open_image(file_path).load()  # Convert to NumPy array
        white_text.config(text=f"Uploaded: {white_raw.shape}", fg='green')


def dark_white_correction():
    global white_raw, dark_raw, spec_img, dw_correction

    # Ensure inputs are NumPy arrays
    white_av = np.mean(white_raw, axis=(0, 1), keepdims=True)  # Average over spatial dimensions
    dark_av = np.mean(dark_raw, axis=(0, 1), keepdims=True)

    # Ensure shape compatibility by broadcasting
    lower_part = white_av - dark_av  # Shape: (1, 1, bands)
    lower_part = np.broadcast_to(lower_part, spec_img.shape)  # Match spec_img shape

    # Perform dark and white correction
    spectral_cube = (spec_img - dark_av) / lower_part  # Apply correction

    # Handle division by zero or near-zero values in-place
    mask = np.isclose(lower_part, 0)  # Shape matches spec_img
    spectral_cube[mask] = 0  # Set to 0 where denominator is 0

    # Replace spec_img with corrected spectral cube
    spec_img = spectral_cube
    # Update UI
    display_image()
    dw_correction.config(text='Corrected', fg="green")


def bright_rgb_correction(val_preview, val):
    global rgb_image
    factor_bright = float(val) / 100  # Normalize the factor between 0 and 2

    print("bright correction")

    # Ensure rgb_image is not None and preview value is different from the actual value
    if rgb_image is not None and val_preview != val:
        rgb_image_corrected = rgb_image.astype(np.float32)  # Convert to float for correction
        rgb_image_corrected = rgb_image_corrected * factor_bright  # Apply brightness factor

        # Clip values to ensure they are within [0, 255] range
        rgb_image_corrected = np.clip(rgb_image_corrected, 0, 255)

        # Convert back to uint8 for display
        rgb_image_corrected = rgb_image_corrected.astype(np.uint8)

        # Display the corrected image
        ax.imshow(rgb_image_corrected)  # Display on the axis
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        image_label.draw()

        print("corrected")


def rotate_spectral_image(image, angle=90):
    """
    Rotate the spectral image by 90 degrees clockwise.

    :param image: NumPy array of shape (height, width, bands)
    :param angle: Rotation angle (only 90, 180, or 270 are valid)
    :return: Rotated spectral image
    """
    if image is None:
        tk.messagebox.showerror("Error", "No spectral image loaded!")
        return None

    if angle not in [90, 180, 270]:
        tk.messagebox.showerror("Error", "Invalid rotation angle! Use 90, 180, or 270 degrees.")
        return image

    # Rotate image while keeping band dimension intact
    rotated_image = np.rot90(image, k=angle // 90, axes=(0, 1))
    return rotated_image


def rotate_image_90():
    """ Rotate the spectral image by 90 degrees and update the display. """
    global spec_img
    spec_img = rotate_spectral_image(spec_img, 90)
    # Refresh displayed image
    new_width = image_info["samples"]
    new_height = image_info["lines"]
    image_info["lines"] = new_width
    image_info["samples"] = new_height
    display_image()
    display_hdr_info(image_info)


def parse_band_selection(bands_str, total_bands):
    selected_bands = set()  # Using a set to prevent duplicates
    try:
        items = bands_str.split(',')
        for item in items:
            item = item.strip()

            if '-' in item:  # Handle range (e.g., "5-10")
                start, end = map(int, item.split('-'))
                selected_bands.update(range(start, end + 1))

            elif '/' in item:  # Handle step-based removal (e.g., "/5")
                step = int(item[1:])  # Get step value (e.g., "5" from "/5")
                selected_bands.update(range(0, total_bands, step))

            else:  # Handle single band (e.g., "3")
                selected_bands.add(int(item))

        # Ensure all indices are within valid range
        selected_bands = {b for b in selected_bands if 0 <= b < total_bands}

        return sorted(selected_bands)  # Return sorted list for consistency

    except ValueError:
        return None


def remove_selected_bands(image, bands_to_remove):
    if image is None:
        messagebox.showerror("Error", "No spectral image loaded!")
        return None

    if not bands_to_remove:
        messagebox.showerror("Error", "No valid bands selected for removal!")
        return image

    bands_to_remove = np.array(bands_to_remove)

    # Check if indices are within the valid range
    if np.any(bands_to_remove >= image.shape[2]) or np.any(bands_to_remove < 0):
        messagebox.showerror("Error", "Invalid band indices!")
        return image

    # Delete the selected bands along the third axis (bands)
    new_image = np.delete(image, bands_to_remove, axis=2)
    image_info["wavelengths"] = np.delete(image_info["wavelengths"], bands_to_remove)
    image_info["bands"] = len(image_info["wavelengths"])
    image_info['default bands'] = None
    return new_image


def remove_bands():
    """ Prompt the user to enter bands or range to remove, then remove them from the spectral image. """
    global spec_img

    if spec_img is None:
        messagebox.showerror("Error", "No spectral image loaded!")
        return

    total_bands = spec_img.shape[2]  # Get total bands count

    # Ask user to input bands or range
    bands_str = simpledialog.askstring("Remove Bands", "Enter band indices or range (e.g., 3,5,7-10,/5):")

    if bands_str:
        bands_to_remove = parse_band_selection(bands_str, total_bands)

        if bands_to_remove is None or len(bands_to_remove) == 0:
            messagebox.showerror("Error", "Invalid input! Use numbers, ranges (e.g., '2,4-7'), or steps (e.g., '/5').")
            return

        # Remove selected bands
        spec_img = remove_selected_bands(spec_img, bands_to_remove)

        # Update the display

        display_hdr_info(image_info)
        display_image()

        messagebox.showinfo("Success", f"Removed bands: {bands_to_remove}")


def app():
    global rectangle_selector
    # File submenu
    my_menu.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label='Open', command=open_file)
    file_menu.add_command(label='Save Cropped Image (RAW/HDR)', command=save_cropped_image_as_raw)
    file_menu.add_command(label='Save file (RAW/HDR)', command=save_image_as_raw)
    # file_menu.add_command(label='Save CSv as...', command=save_csv_area)

    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)

    # Edit submenu
    my_menu.add_cascade(label="Edit", menu=edit_menu)
    edit_menu.add_command(label='Rotate', command=rotate_image_90)
    edit_menu.add_command(label='Remove bands', command=remove_bands)

    # Veiw
    my_menu.add_cascade(label="View", menu=view_menu)
    view_menu.add_command(label='Spectral window', command=open_spectral_window)
    # ------------------------------------------------------------------------------------------

    # Right menu label
    right_menu_label.config(justify="left", background="lightgray")
    right_menu_label.grid(row=0, column=2, rowspan=3, sticky='ns')

    # HDR info text
    hdr_text_box.config(wrap=tk.WORD, width=30, height=25, bg='lightgray', state=tk.DISABLED)
    hdr_Info_Label_scrollbar.config(orient=tk.VERTICAL, command=hdr_text_box.yview)
    hdr_Info_Label_scrollbar.grid(row=0, column=0, columnspan=5, sticky='nes')
    hdr_text_box.config(yscrollcommand=hdr_Info_Label_scrollbar.set)
    hdr_text_box.grid(row=0, column=0, sticky='new', columnspan=5)

    # ComboBoxes for RGB bands
    red_combo_box.grid(row=1, column=1, sticky='nw')
    green_combo_box.grid(row=2, column=1, sticky='nw')
    blue_combo_box.grid(row=3, column=1, sticky='nw')

    # Update bands button
    update_bands_button = tk.Button(right_menu_label, text='Update\nbands', command=display_image, width=7, height=4)
    update_bands_button.grid(row=1, rowspan=3, column=0, sticky='nw')

    # Rectangle selector-----------------------------------------------------------------------------------
    rectangle_selector = RectangleSelector(
        ax,
        on_select,
        useblit=False,  # Set to False to keep the rectangle visible
        button=[1],  # Only respond to left mouse button
        minspanx=5, minspany=5,  # Minimum size for selection
        interactive=True  # Allows resizing and moving the selection
    )

    # rectangle selector size enter
    rectangle_size_y.config(width=6)
    rectangle_size_x.config(width=6)
    rectangle_size_x.insert(0, '0')
    rectangle_size_y.insert(0, '0')
    tk.Label(right_menu_label, text="X", bg='lightgray').grid(row=4, column=0, columnspan=1, pady=3, padx=10,
                                                              sticky='e')

    rectangle_size_x.grid(row=4, column=0, columnspan=2, sticky='nw', pady=2)
    rectangle_size_y.grid(row=4, column=1, columnspan=2, sticky='w', pady=2)
    update_bands_button = tk.Button(right_menu_label, text='Set', command=set_rectangle_size, bg='lightgray')
    update_bands_button.grid(row=4, column=1, sticky='e', padx=40, pady=0)
    # ---------------------------------------------------------------------------------------------------------
    # Chooose single spectral
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Checkboxes of build spectral
    Checkbutton(right_menu_label, text='Build single spectral', variable=checker_single_block, bg='lightgray').grid(
        row=5, column=0, columnspan=2, sticky='nw')
    Checkbutton(right_menu_label, text='Add new spector', variable=checker_add_new_spec, bg='lightgray').grid(row=6,
                                                                                                              column=0,
                                                                                                              columnspan=2,
                                                                                                              sticky='nw')

    # Button Clear spectral
    update_bands_button = tk.Button(right_menu_label, text='Clear spectral', command=clear_spectral_plot,
                                    bg='lightgray', width=12, height=1)
    update_bands_button.grid(row=7, column=0, columnspan=2, sticky='nw')

    # Button plot rectangle
    update_bands_button = tk.Button(right_menu_label, text='Plot rectangle', command=plot_rectangle_spectral,
                                    bg='lightgray', width=12, height=1)
    update_bands_button.grid(row=8, column=0, columnspan=2, sticky='nw')

    # Dark and wight correction
    tk.Label(right_menu_label, text="Dark and Wight correction", bg='lightgray').grid(row=9, column=0, columnspan=2,
                                                                                      pady=5, sticky='nw')
    update_bands_button = tk.Button(right_menu_label, text='Dark', command=open_dark_raw, bg='lightgray', width=6,
                                    height=1)
    update_bands_button.grid(row=10, column=0, sticky='nw')
    dark_text.grid(row=10, column=1, columnspan=2, sticky='nw')
    update_bands_button = tk.Button(right_menu_label, text='White', command=open_white_raw, bg='lightgray', width=6,
                                    height=1)
    update_bands_button.grid(row=11, column=0, sticky='nw')
    white_text.grid(row=11, column=1, columnspan=2, sticky='nw')
    update_bands_button = tk.Button(right_menu_label, text='Correction', command=dark_white_correction, bg='lightgray',
                                    width=6, height=1)
    update_bands_button.grid(row=12, column=0, sticky='nw')
    dw_correction.grid(row=12, column=1, columnspan=2, sticky='nw')

    # Add a slider to control brightness (scale from 0 to 300 for 0 to 3 factor range)
    brightness_slider = tk.Scale(right_menu_label, from_=1, to=500, orient=tk.HORIZONTAL, label="Brightness")
    brightness_slider.set(100)  # Set the initial value to 100 (no change)
    brightness_slider_preview_value = brightness_slider.get()
    brightness_slider.grid(row=13, column=0, columnspan=2, sticky='nwe')

    # Update the image when the slider is moved
    brightness_slider.bind("<ButtonRelease-1>", lambda event: bright_rgb_correction(brightness_slider_preview_value,
                                                                                    brightness_slider.get()))

    # loop
    root.mainloop()


# Main application
if __name__ == "__main__":
    app()
