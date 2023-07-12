import tkinter as tk
from tkinter import messagebox, ttk, filedialog, Toplevel, Label, Scale, scrolledtext
import numpy as np
from matplotlib import colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ttkbootstrap import Style
from Filter import run_fit
import pandas as pd
import matplotlib.pyplot as plt
import function_analysis
import threading
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import distance_transform_edt


def create_app():
    def create_detector_time_map_ui(fit_x, fit_y, pixel_size, sigma, delta_time, progress_callback=None):
        # Determine the range of the original coordinates
        nonlocal image_window, ax_time_map
        highlight = []
        highlight_points = []
        gen_x_pixel = None
        gen_y_pixel = None
        gen_time = None

        if image_window is not None:
            image_window.destroy()

        def on_listbox_select(event):
            nonlocal gen_x_pixel, gen_y_pixel, gen_time, ax_time_map, highlight

            # Check if the variables have been initialized
            if gen_x_pixel is None or gen_y_pixel is None or gen_time is None:
                return

            # Clear the previous scatter plot
            if highlight:
                for h in highlight:
                    h.remove()
                highlight = []

            # Get the current selection from the Listbox
            selected = treeview.selection()
            if selected:  # If there is a selection
                # Get the values of the selected row
                x_str, y_str, t_str, weight_str = treeview.item(selected, "values")
                x = float(x_str)
                y = float(y_str)
                if t_str:  # Check if t_str is not empty
                    t_values = list(map(float, t_str.split(', ')))
                else:
                    t_values = [0]

                # Highlight the selected point
                for t in t_values:
                    h = ax_time_map.scatter(x, y, c='magenta', s=500,
                                            marker='*')  # Use a larger size (s), a different color (c), and a star marker
                    highlight.append(h)

                # Redraw the figure
                canvas.draw()
            else:
                print("No selection")

        def load_data():
            nonlocal gen_x_pixel, gen_y_pixel, gen_time, ax_time_map, highlight_points
            # Remove the old points
            if highlight_points:
                for p in highlight_points:
                    p.remove()
                highlight_points = []

            file_path = filedialog.askopenfilename(filetypes=[('All Files', '*.*'), ('Data Files', '*.dat')])
            if file_path:
                df = pd.read_csv(file_path, header=None, sep=" ")
                parameters_data, data = df.iloc[0], df.iloc[1:]

                f2, f1, _, _, _, _, _, _ = [float(val[1:]) if isinstance(val, str) and val.startswith('#') else val for
                                            val in parameters_data.values]  # to remove "#" before f1
                gen_x, gen_y = data[0].values.astype(float), data[1].values.astype(float)

                # Calculate the pixel coordinates of the newly loaded data
                gen_x_pixel = np.floor((gen_x - x_min) / pixel_size).astype(int)
                gen_y_pixel = np.floor((gen_y - y_min) / pixel_size).astype(int)

                # Ensure that the pixel coordinates are within the valid range
                gen_x_pixel = np.clip(gen_x_pixel, 0, time_map.shape[1] - 1)
                gen_y_pixel = np.clip(gen_y_pixel, 0, time_map.shape[0] - 1)

                # Extract the time from the time map
                gen_time = [time_map[y, x, :time_counter[y, x]] for y, x in zip(gen_y_pixel, gen_x_pixel)]

                # Extract the weight from the weight map
                gen_weight = [weight_map[y, x] for y, x in zip(gen_y_pixel, gen_x_pixel)]

                # Add the points to the time map heatmap
                points = ax_time_map.scatter(gen_x_pixel, gen_y_pixel, c='red', s=100)
                highlight_points.append(points)

                canvas.draw()

                # Clear the treeview widget
                treeview.delete(*treeview.get_children())

                # Add the times, corresponding x, y points, and weights to the Treeview widget
                with open('map_result.dat', 'w') as f:
                    for x, y, times, weight in zip(gen_x_pixel, gen_y_pixel, gen_time, gen_weight):
                        times_str = ", ".join(str(t) for t in times)
                        treeview.insert('', 'end',
                                        values=(x, y, times_str, weight))  # Add each item to the end of the Treeview
                        f.write(f"{x}, {y}, [{times_str}], {weight}\n")  # Write the result to the file

        larger_map = 0.2  # see definition below
        max_times_per_pixel = 100
        threshold = 0.1  # For  the weight map

        # Determine the range of the original coordinates
        x_min, x_max = np.min(fit_x) - larger_map, np.max(fit_x) + larger_map
        y_min, y_max = np.min(fit_y) - larger_map, np.max(fit_y) + larger_map

        detector_width = int(np.ceil((x_max - x_min) / pixel_size))
        detector_height = int(np.ceil((y_max - y_min) / pixel_size))

        progress_increment = 50.0 / 1000  # The range of the progress bar for this function is from 50% to 100%

        # Initialize the counts array
        counts = np.zeros((detector_height, detector_width))  # Note the reversed order

        time_map = np.full((detector_height, detector_width, max_times_per_pixel), np.nan)
        time_counter = np.zeros((detector_height, detector_width), dtype=int)

        # Create a binary image representing the trajectory
        trajectory_image = np.zeros((detector_height, detector_width))
        for x, y in zip(fit_x, fit_y):
            x_pixel = int(np.floor((x - x_min) / pixel_size))
            y_pixel = int(np.floor((y - y_min) / pixel_size))
            trajectory_image[y_pixel, x_pixel] = 1

        # Compute the distance transform
        dist = distance_transform_edt(1 - trajectory_image)

        # Compute the weight map from the distance transform
        weight_map = np.exp(-dist / sigma)

        # Count the number of fitted points at each pixel
        for i, (x, y) in enumerate(zip(fit_x, fit_y)):
            x_pixel = int(np.floor((x - x_min) / pixel_size))
            y_pixel = int(np.floor((y - y_min) / pixel_size))
            counts[y_pixel, x_pixel] += 1

            # Check if the weight is above a threshold
            if weight_map[y_pixel, x_pixel] > threshold * np.max(weight_map):
                # Copy time information to additional pixels
                for dx in range(-delta_time, delta_time + 1):
                    for dy in range(-delta_time, delta_time + 1):
                        new_x_pixel = x_pixel + dx
                        new_y_pixel = y_pixel + dy

                        # Check if the additional pixel is within the detector boundaries
                        if 0 <= new_x_pixel < detector_width and 0 <= new_y_pixel < detector_height:
                            if time_counter[new_y_pixel, new_x_pixel] < max_times_per_pixel:
                                time_map[new_y_pixel, new_x_pixel, time_counter[new_y_pixel, new_x_pixel]] = i
                                time_counter[new_y_pixel, new_x_pixel] += 1

            if progress_callback is not None:
                progress_callback(increment=progress_increment)

        # Create a new window to display the image
        image_window = tk.Toplevel()
        image_window.title("Detector Time Map")
        image_window.geometry("800x600")  # Set the initial size of the window
        image_window.grid_rowconfigure(0, weight=1)
        image_window.grid_columnconfigure(0, weight=1)

        # Create a new figure
        fig = plt.figure(figsize=(18, 6))

        # Add the 3D subplot
        ax_map = fig.add_subplot(141, projection='3d')  # Change this line

        # Plot the surface
        X, Y = np.meshgrid(np.arange(detector_width), np.arange(detector_height))
        ax_map.plot_surface(X, Y, counts, cmap='viridis')

        # Plot the points where ntime = 1
        mask_ntime_1 = counts == 1
        ax_map.scatter(X[mask_ntime_1], Y[mask_ntime_1], counts[mask_ntime_1], color='blue')

        # Plot the points where ntime > 1
        mask_ntime_gt_1 = counts > 1
        ax_map.scatter(X[mask_ntime_gt_1], Y[mask_ntime_gt_1], counts[mask_ntime_gt_1], color='red')

        ax_map.set_title('Detector Time Map')
        ax_map.set_xlabel('X')
        ax_map.set_ylabel('Y')
        ax_map.set_zlabel('N-times')

        # Add the counts heatmap
        ax_counts = fig.add_subplot(142)  # Change this line
        cax1 = ax_counts.imshow(counts, cmap='viridis', interpolation='nearest', origin='lower')
        fig.colorbar(cax1, ax=ax_counts)
        ax_counts.set_title('Counts')

        # Add the weight map heatmap
        ax_weight_map = fig.add_subplot(143)  # Add this line
        cax3 = ax_weight_map.imshow(weight_map, cmap='viridis', interpolation='nearest',
                                    origin='lower')  # Add this line
        fig.colorbar(cax3, ax=ax_weight_map)  # Add this line
        ax_weight_map.set_title('Weight Map')  # Add this line

        # Add the time map heatmap
        ax_time_map = fig.add_subplot(144)  # Change this line
        first_times = time_map[:, :, 0]  # Extract the first time at each pixel
        cax2 = ax_time_map.imshow(first_times, cmap='viridis', interpolation='nearest', origin='lower')
        fig.colorbar(cax2, ax=ax_time_map)
        ax_time_map.set_title('Time Map')

        # Add the canvas to the window
        canvas = FigureCanvasTkAgg(fig, master=image_window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)

        # Create the "Load Data" button
        load_button = tk.Button(master=image_window, text="Load Data", command=load_data, width=15, height=2)
        load_button.grid(row=1, column=0, sticky='w', padx=200)

        # Create the Treeview widget for displaying time values and weights
        treeview = ttk.Treeview(master=image_window, columns=("X", "Y", "Time", "Weight"), show="headings")

        # Set the column headings and alignments
        treeview.heading("X", text="X", anchor='center')
        treeview.heading("Y", text="Y", anchor='center')
        treeview.heading("Time", text="Time", anchor='center')
        treeview.heading("Weight", text="Weight", anchor='center')

        # Set the column alignments
        treeview.column("X", anchor='center')
        treeview.column("Y", anchor='center')
        treeview.column("Time", anchor='center')
        treeview.column("Weight", anchor='center')

        treeview.bind('<<TreeviewSelect>>',
                      on_listbox_select)  # Bind the selection event to the on_listbox_select function
        treeview.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=150, pady=15)

    def display_parameters(original_params, fitted_params):

        nonlocal param_window

        if param_window is not None:
            param_window.destroy()

        param_window = tk.Toplevel()
        param_window.title("Data and Plot")

        data_frame = tk.Frame(param_window)
        data_frame.pack(side="left", padx=10, pady=10, expand=True, fill=tk.BOTH)

        tree = ttk.Treeview(data_frame)

        tree["columns"] = ("Original Value", "Fitted Value")
        tree.column("#0", width=150)
        tree.column("Original Value", width=150)
        tree.column("Fitted Value", width=150)

        tree.heading("#0", text="Parameter")
        tree.heading("Original Value", text="Original Value")
        tree.heading("Fitted Value", text="Fitted Value")

        param_names = ['A_x1', 'A_x2', 'B_y1', 'B_y2', 'f1', 'f2', 'p_x1', 'p_y1', 'p_x2', 'p_y2']
        fitted_params[5] = fitted_params[5] / (2 * np.pi)
        fitted_params[4] = fitted_params[4] / (2 * np.pi)

        for i, param_name in enumerate(param_names):
            original_value = original_params.get(param_name, 0)
            fitted_value = fitted_params[i] if i < len(fitted_params) else "N/A"
            # Convert the value to pi unit if parameter is phase
            if 'p_' in param_name:
                original_value = round(original_value / np.pi, 2)
                fitted_value = round(fitted_value / np.pi, 2) if isinstance(fitted_value, (int, float)) else "N/A"
            tree.insert("", "end", text=param_name,
                        values=(f"{original_value}π" if 'p_' in param_name else original_value,
                                f"{fitted_value}π" if 'p_' in param_name else fitted_value))

        tree.pack(expand=True, fill=tk.BOTH)

    def redraw_on_scale_change(*args):
        # Check if data is loaded
        if data_loaded:
            if auto_scale_var.get() == 1:
                ax.set_aspect('equal', 'box')
            else:
                ax.set_aspect('auto')  # Reset to the default aspect
            draw_plot(ax, canvas, gen_x, gen_y, 'Loaded Data', 'Loaded data', scatter=True)

    def open_limit_window():
        limit_window = tk.Toplevel(app)
        limit_window.title("Set Parameter Factors")

        # update the value of the bounds_factor dictionary
        def update_bounds_factor(param, val):
            bounds_factor_dict[param] = round(float(val), 2)
            # Update the text displayed on the label
            bounds_factor_labels[param].config(text=f"Factor for {param}: {bounds_factor_dict[param]}")

        # Initialize bounds_factor_dict with the current factor values
        for param in params.keys():
            if param == "n":  # Skip the parameter "n"
                continue
            bounds_factor_dict[param] = 0.5  # Set initial factor to 0.5

        # Create a dictionary to hold the labels for each parameter
        bounds_factor_labels = {}
        for i, param in enumerate(params.keys()):
            if param == "n":  # Skip the parameter "n"
                continue
            bounds_factor_labels[param] = tk.Label(limit_window,
                                                   text=f"Factor for {param}: {bounds_factor_dict[param]}")
            bounds_factor_labels[param].grid(row=i, column=0, padx=3, pady=10, sticky='w')

            factor_scale = tk.Scale(limit_window, from_=0, to=1, resolution=0.01, orient='horizontal',
                                    command=lambda val, p=param: update_bounds_factor(p, val))
            factor_scale.grid(row=i, column=1, padx=(0, 5), pady=(0, 10), sticky='e')
            factor_scale.set(bounds_factor_dict[param])  # Set the initial factor to 0.5

    def show_logs():

        log_window = tk.Toplevel(app)
        log_window.title("History logs")

        txt = scrolledtext.ScrolledText(log_window, undo=True)
        txt['font'] = ('consolas', '12')
        txt.pack(expand=True, fill='both')

        # Load log file
        with open('filter_log.txt', 'r') as log_file:
            log_contents = log_file.read()

        # Insert log contents to the text widget
        txt.insert(tk.INSERT, log_contents)

    def draw_plot(ax, canvas, x, y, title, label, clear=True, scatter=True):
        if clear:
            ax.clear()
        if scatter:
            ax.scatter(x, y, label=label)
        else:
            ax.plot(x, y, color='red', label=label)
        ax.set_title(title)
        ax.legend()
        if auto_scale_var.get() == 1:
            ax.set_aspect('equal', 'box')
        else:
            ax.set_aspect('auto')  # Reset to the default aspect
        canvas.draw()

    def load_file():
        nonlocal gen_x, gen_y, data_loaded, params, parameters_loaded, t_measured, previous_window, bounds_factor_dict
        file_path = filedialog.askopenfilename(filetypes=[('All Files', '*.*'), ('Data Files', '*.dat')])
        if file_path:
            df = pd.read_csv(file_path, header=None, sep=" ")
            parameters_data, data = df.iloc[0], df.iloc[1:]

            f2, f1, _, _, _, _, _, _ = [float(val[1:]) if isinstance(val, str) and val.startswith('#') else val for val
                                        in parameters_data.values]  # to remove "#" before f1
            gen_x, gen_y = data[0].values, data[1].values

            if previous_window is not None:
                previous_window.destroy()

            # Only keep one period of data
            gen_x, gen_y = function_analysis.keep_one_period(gen_x, gen_y)

            gen_x = gen_x.astype(float)
            gen_y = gen_y.astype(float)
            # This part is prepared for the more complex x,y data
            # by performing an FFT on them and obtaining the frequency, amplitude
            analysis_results = function_analysis.xy_fft(gen_x, gen_y)

            # Assign the returned result to a variable
            A_x = analysis_results["gen_x_amplitudes"]
            p_x = analysis_results["gen_x_phases"]
            B_y = analysis_results["gen_y_amplitudes"]
            p_y = analysis_results["gen_y_phases"]
            # f_x = analysis_results["gen_x_frequencies"]
            # f_y = analysis_results["gen_y_frequencies"]

            # Process the data to find phase difference
            processed_data = function_analysis.process_data(gen_x, gen_y, f1, f2)

            f1 = processed_data["f1"]
            f2 = processed_data["f2"]

            if auto_scale_var.get() == 1:
                ax.set_aspect('equal', 'box')
            else:
                ax.set_aspect('auto')  # Reset to the default aspect

            # params = {'A': A[0], 'B': B[0], 'w1': w1, 'w2': w2, 'p1': 0, 'p2': phase_difference_in_pi * np.pi,
            #           'n': len(gen_x)}

            params = {
                'f1': f1,
                'f2': f2,
                'n': len(gen_x),
            }

            # 为 gen_x 的每个分量动态添加A和p到params字典
            for i, (Ax, px) in enumerate(zip(A_x, p_x), 1):
                params[f'A_x{i}'] = Ax
                params[f'p_x{i}'] = px

            # 为 gen_y 的每个分量动态添加A和p到params字典
            for i, (By, py) in enumerate(zip(B_y, p_y), 1):
                params[f'B_y{i}'] = By
                params[f'p_y{i}'] = py

            draw_plot(ax, canvas, gen_x, gen_y, 'Loaded data', 'Loaded data', clear=True, scatter=True)

            # Marked every 50 points in the loaded data
            for i in range(0, len(gen_x), 20):
                x = gen_x[i]
                y = gen_y[i]
                label = f'({x:.2f}, {y:.2f}, {i})'
                ax.annotate(label, (x, y), xytext=(5, -10), textcoords='offset points', ha='left', va='top')
                ax.plot(x, y, 'ro', markersize=5)  # Highlighted points are red circles

            canvas.draw()

            parameters_loaded = data_loaded = True

            update_status_label()

            xy_window = tk.Toplevel()
            xy_window.title("Data and Plot")

            data_frame = tk.Frame(xy_window)
            data_frame.pack(side="left", padx=10, pady=10, expand=True, fill=tk.BOTH)

            tree = ttk.Treeview(data_frame)

            tree["columns"] = ("Value",)

            tree.column("#0", width=150)
            tree.column("Value", width=150)

            tree.heading("#0", text="Parameter")
            tree.heading("Value", text="Value")

            for param_name, param_value in params.items():
                tree.insert("", "end", text=param_name, values=(param_value,))

            tree.pack(expand=True, fill=tk.BOTH)

            plot_frame = tk.Frame(xy_window)
            plot_frame.pack(side="right", padx=10, pady=10, expand=True, fill=tk.BOTH)

            fig, axs = plt.subplots(2, 1, figsize=(6, 6))

            axs[0].plot(gen_x, label='gen_x')
            axs[0].set_title('gen_x')
            axs[0].set_xlabel('Index')
            axs[0].set_ylabel('Value')
            axs[0].legend()

            axs[1].plot(gen_y, label='gen_y')
            axs[1].set_title('gen_y')
            axs[1].set_xlabel('Index')
            axs[1].set_ylabel('Value')
            axs[1].legend()

            plt.tight_layout()

            canvas_plot = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack(expand=True, fill=tk.BOTH)

            previous_window = xy_window
            bounds_factor_dict = {param_name: 0.5 for param_name in params}

            xy_window.update()

        if data_loaded:
            data_status_label.config(text="Data: Loaded", background="green")
            button_set_limit['state'] = 'normal'

        if parameters_loaded:
            parameters_status_label.config(text="Parameters: Loaded", background="green")

    def update_status_label():
        if data_loaded and parameters_loaded:
            filter_button['state'] = 'normal'
            button_set_limit['state'] = 'normal'

    # Checkbutton state change callback
    def redraw_on_check():
        Fit()

    def run_fit_in_thread(x, y, params, filter_press_count, progress_var, progress_window, status_label,pixel_size, sigma, delta_time):

        try:
            def progress_callback(xk, convergence, progress_range=(0, 50)):
                # We divide the progress bar into two parts, this callback function is used to update the first half
                current_progress = progress_var.get()
                progress_increment = (progress_range[1] - progress_range[0]) / 50  # 假设我们有50步
                progress_var.set(min(current_progress + progress_increment, progress_range[1]))

            def progress_callback_2(increment):
                current_progress = progress_var.get()
                progress_var.set(min(current_progress + increment, 100))

            # Set the labelled text to the fitting process
            status_label.config(text="Fitting in progress...")

            fit_results = run_fit(x, y, params, bounds_factor_dict, filter_press_count,
                                  progress_callback=progress_callback)
            fitted_params = fit_results["fitted_params"]

            status_label.config(text="Mapping...")
            # Create the detector time map
            create_detector_time_map_ui(fit_results["x_fit"], fit_results["y_fit"], pixel_size, sigma, delta_time,
                                        progress_callback=progress_callback_2)

            def update_ui():
                display_parameters(params, fitted_params)

                fit_x2 = fit_results["x_fit"]
                fit_y2 = fit_results["y_fit"]

                ax.clear()
                if check_var1.get() == 1:
                    draw_plot(ax, canvas, gen_x, gen_y, 'Original and Fitted Data', 'Original data', clear=False,
                              scatter=True)
                if check_var2.get() == 1:
                    draw_plot(ax, canvas, fit_x2, fit_y2, 'New Data', 'Filtered data', clear=False, scatter=False)
                    # 标记每过50个点的节点
                    for i in range(0, len(fit_x2), 50):
                        label = f'({fit_x2[i]:.2f}, {fit_y2[i]:.2f}, {i})'  # 标记文本包含了x、y和索引
                        ax.annotate(label, (fit_x2[i], fit_y2[i]), xytext=(5, -10),
                                    textcoords='offset points', ha='left', va='top')
                        ax.plot(fit_x2[i], fit_y2[i], 'ro', markersize=5)  # 高亮标记的点为红色圆圈

                canvas.draw()

                progress_window.destroy()

            app.after(0, update_ui)

        except Exception as e:
            tk.messagebox.showerror("Error", str(e))

    def Fit(pixel_size, sigma, delta_time):
        nonlocal gen_x, gen_y, params, filter_press_count, new_window

        if new_window is not None:
            new_window.destroy()

        filter_press_count += 1  # increment counter

        progress_window = tk.Toplevel()
        progress_window.title("Fitting Progress")

        # 创建一个变量来存储进度值
        progress_var = tk.DoubleVar()
        progress_var.set(0)

        # 创建一个标签来显示状态信息
        status_label = tk.Label(progress_window, text="Initializing...", font=("Arial", 12))
        status_label.pack()

        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=300)
        progress_bar.pack(padx=20, pady=20)

        # Start a new thread to run the time-consuming fitting operation
        fit_thread = threading.Thread(target=run_fit_in_thread,
                                      args=(gen_x, gen_y, params, filter_press_count, progress_var, progress_window,
                                            status_label,pixel_size, sigma, delta_time))
        fit_thread.daemon = True  # Set as a daemon thread so that when the main program exits the thread will also exit
        fit_thread.start()

    def on_fit_button_clicked():
        def validate_inputs(*args):
            try:
                pixel_size = float(pixel_size_var.get())
                sigma = int(sigma_var.get())
                delta_time = int(delta_time_var.get())

                if 0.005 <= pixel_size <= 0.1 and 1 <= sigma <= 50 and 1 <= delta_time <= 15:
                    confirm_button.config(state='normal')
                else:
                    confirm_button.config(state='disabled')
            except ValueError:
                confirm_button.config(state='disabled')

        # Create a new window
        settings_window = tk.Toplevel()
        settings_window.title("Settings")

        # Create variables
        pixel_size_var = tk.StringVar(value="0.01")  # Default value
        sigma_var = tk.StringVar(value="10")  # Default value
        delta_time_var = tk.StringVar(value="5")  # Default value

        # Create labels and text inputs
        tk.Label(settings_window, text="Pixel Size (recommended: 0.005 to 0.1)").grid(row=0, column=0)
        pixel_size_entry = tk.Entry(settings_window, textvariable=pixel_size_var)
        pixel_size_entry.grid(row=0, column=1)
        pixel_size_var.trace('w', validate_inputs)

        tk.Label(settings_window, text="Sigma (recommended: 1 to 50)").grid(row=1, column=0)
        sigma_entry = tk.Entry(settings_window, textvariable=sigma_var)
        sigma_entry.grid(row=1, column=1)
        sigma_var.trace('w', validate_inputs)

        tk.Label(settings_window, text="Delta Time (recommended: 1 to 15)").grid(row=2, column=0)
        delta_time_entry = tk.Entry(settings_window, textvariable=delta_time_var)
        delta_time_entry.grid(row=2, column=1)
        delta_time_var.trace('w', validate_inputs)

        # Create a confirm button
        confirm_button = tk.Button(settings_window, text="Confirm",
                                   command=lambda: on_confirm_button_clicked(float(pixel_size_var.get()),
                                                                             int(sigma_var.get()),
                                                                             int(delta_time_var.get()),
                                                                             settings_window))
        confirm_button.grid(row=3, column=0, columnspan=2)
        confirm_button.config(state='disabled')

        validate_inputs()  # Initially validate the inputs

    def on_confirm_button_clicked(pixel_size, sigma, delta_time, settings_window):
        settings_window.destroy()
        Fit(pixel_size, sigma, delta_time)

    # Initialize
    data_loaded = False
    parameters_loaded = False
    filter_press_count = 0  # initialize counter at global scope
    previous_window = None
    gen_x, gen_y, t_measured = None, None, None
    params = None
    new_window = None
    param_window = None
    image_window = None

    ax_time_map = None  # Initialize ax_time_map
    # Initial value 0.05
    # 定义空词典
    bounds_factor_dict = dict()

    app = tk.Tk()
    app.title("Fitting Tool")
    app.geometry("800x600")

    # Configure row and column weights
    for i in range(12):
        app.rowconfigure(i, weight=1)

    # Increase the weight of the column where the graph will be placed
    app.columnconfigure(0, weight=1)
    app.columnconfigure(1, weight=3)
    app.columnconfigure(2, weight=3)

    fig = Figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111)

    # Create the canvas as a child of a Frame
    frame = tk.Frame(app)
    frame.grid(row=0, column=1, rowspan=12, columnspan=2, padx=5, pady=5, sticky='nsew')
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # Check buttons
    check_var1 = tk.IntVar(value=1)  # set initial to true
    check_var2 = tk.IntVar(value=1)
    auto_scale_var = tk.IntVar()

    checks_frame = ttk.Frame(app)
    checks_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

    check_button1 = ttk.Checkbutton(checks_frame, text="Show Original Data", variable=check_var1)
    check_button1.pack(side='top', padx=5, pady=5)

    check_button2 = ttk.Checkbutton(checks_frame, text="Show Fitted Data", variable=check_var2)
    check_button2.pack(side='top', padx=5, pady=5)

    auto_scale_checkbox = ttk.Checkbutton(checks_frame, text="Auto Scale", variable=auto_scale_var)
    auto_scale_checkbox.pack(side='top', padx=5, pady=5)
    auto_scale_var.trace('w', redraw_on_scale_change)

    ifixb_dict = {param: tk.IntVar() for param in ["A", "B", "w1", "w2", "p1", "p2"]}
    for param in ifixb_dict:
        ifixb_dict[param].set(1)  # All checkboxes are checked by default (ifxib all set to 1)

    # Operation buttons
    operations_frame = ttk.Frame(app)
    operations_frame.grid(row=2, column=0, padx=10, pady=5, sticky='ew')

    button_set_limit = ttk.Button(operations_frame, text="Set Bounds", command=open_limit_window, state='disabled')
    button_set_limit.pack(side='top', padx=5, pady=5, fill='x')

    log_button = ttk.Button(operations_frame, text="Log", command=show_logs)
    log_button.pack(side='top', padx=5, pady=5, fill='x')

    # Status frame
    status_frame = ttk.Frame(app)
    status_frame.grid(row=3, column=0, padx=5, pady=5, sticky='ew')

    data_status_label = ttk.Label(status_frame, text="Data: Not Loaded", background="red")
    data_status_label.pack(side="top", fill="x", expand=True, padx=5, pady=5)

    parameters_status_label = ttk.Label(status_frame, text="Parameters: Not Loaded", background="red")
    parameters_status_label.pack(side="top", fill="x", expand=True, padx=5, pady=5)

    # Attach the callback to the Checkbuttons
    check_var1.trace('w', lambda *args: redraw_on_check())
    check_var2.trace('w', lambda *args: redraw_on_check())

    # Buttons
    buttons_frame = ttk.Frame(app)
    buttons_frame.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

    filter_button = ttk.Button(buttons_frame, text="Fit", command=on_fit_button_clicked, state='disabled')
    filter_button.pack(side='top', padx=10, pady=10, fill='x')

    load_button = ttk.Button(buttons_frame, text="Load Data & Parameters", command=load_file)
    load_button.pack(side='top', padx=10, pady=10, fill='x')

    app.mainloop()


if __name__ == '__main__':
    create_app()
