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
from matplotlib.colors import PowerNorm
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
import ast
from matplotlib.colors import PowerNorm


def create_app():
    def create_detector_time_map_ui(fit_x, fit_y, pixel_size, sigma, delta_time, progress_callback=None):
        # Determine the range of the original coordinates
        nonlocal image_window, ax_time_map
        highlight = []
        highlight_points = []
        gen_x_pixel = None
        gen_y_pixel = None
        gen_time = None
        zoomed_circle = None
        if image_window is not None:
            image_window.destroy()

        def on_cluster_listbox_select(event):
            nonlocal gen_x_pixel, gen_y_pixel, gen_time, ax_time_map, highlight, zoom_window, zoomed_circle

            # Remove old zoom window
            if zoom_window is not None:
                zoom_window.destroy()

            # Check if the variables have been initialized
            if gen_x_pixel is None or gen_y_pixel is None or gen_time is None:
                return

            # Clear the previous scatter plot
            if highlight:
                for h in highlight:
                    h.remove()
                highlight = []

            # Get the current selection from the cluster_treeview
            selected = cluster_treeview.selection()
            if selected:  # If there is a selection
                # Get the values of the selected row
                x_str, y_str, t_str, weight_str, weight_time_str = cluster_treeview.item(selected, "values")
                x = round(float(x_str))
                y = round(float(y_str))
                # Replace 'nan' with 'None'
                t_str = t_str.replace('nan', 'None')
                # Check if t_str and weight_str are not empty
                t_values = ast.literal_eval(t_str) if t_str else [0]

                # Destroy the previous zoom window if it exists
                if zoom_window is not None:
                    zoom_window.destroy()

                # Create a new window to display the zoomed view
                zoom_window = tk.Toplevel()
                zoom_window.title("Zoomed View")
                zoom_window.geometry("600x400")  # Set the initial size of the window

                # Create the new figure
                fig_zoom, ax_zoom = plt.subplots()

                # Plot the time map heatmap

                zoom_size = 20  # Change the size as needed
                zoomed_first_times = first_times[max(0, y - zoom_size):min(first_times.shape[0], y + zoom_size),
                                     max(0, x - zoom_size):min(first_times.shape[1], x + zoom_size)]
                ax_zoom.imshow(zoomed_first_times, cmap='viridis', interpolation='nearest', origin='lower')

                # Highlight the selected point
                ax_zoom.scatter(zoom_size, zoom_size, c='red')

                # Highlight the selected point
                for t in t_values:
                    h = ax_time_map.scatter(x, y, c='magenta', s=200, marker='*')
                    highlight.append(h)

                # Remove the previous circle if it exists
                if zoomed_circle is not None:
                    zoomed_circle.remove()

                # Draw a circle around the zoomed area
                zoomed_circle = plt.Circle((x, y), zoom_size + 10, fill=False, color='red', linestyle='dashed')
                ax_time_map.add_patch(zoomed_circle)

                # Add the canvas to the window
                canvas_zoom = FigureCanvasTkAgg(fig_zoom, master=zoom_window)
                canvas_zoom.draw()
                canvas_zoom.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                # Redraw the figure
                canvas.draw()

            else:
                print("No selection")

        def on_listbox_select(event):
            nonlocal gen_x_pixel, gen_y_pixel, gen_time, ax_time_map, highlight, zoom_window, zoomed_circle

            # Remove old zoom window
            if zoom_window is not None:
                zoom_window.destroy()

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
                x_str, y_str, t_str, weight_str, weight_time_str = treeview.item(selected, "values")
                x = int(x_str)
                y = int(y_str)
                # Replace 'nan' with 'None'
                t_str = t_str.replace('nan', 'None')
                # Check if t_str and weight_str are not empty
                t_values = [ast.literal_eval(t_str)] if isinstance(t_str, int) else ast.literal_eval(t_str)
                weight_values = ast.literal_eval(weight_str) if weight_str else [0]

                # Destroy the previous zoom window if it exists
                if zoom_window is not None:
                    zoom_window.destroy()

                # Create a new window to display the zoomed view
                zoom_window = tk.Toplevel()
                zoom_window.title("Zoomed View")
                zoom_window.geometry("600x400")  # Set the initial size of the window

                # Create the new figure
                fig_zoom, ax_zoom = plt.subplots()

                # Plot the time map heatmap

                zoom_size = 20  # Change the size as needed
                zoomed_first_times = first_times[max(0, y - zoom_size):min(first_times.shape[0], y + zoom_size),
                                     max(0, x - zoom_size):min(first_times.shape[1], x + zoom_size)]
                ax_zoom.imshow(zoomed_first_times, cmap='viridis', interpolation='nearest', origin='lower')

                # Highlight the selected point
                ax_zoom.scatter(zoom_size, zoom_size, c='red')

                # Highlight the selected point
                for t in t_values:
                    h = ax_time_map.scatter(x, y, c='magenta', s=200, marker='*')
                    highlight.append(h)

                # Remove the previous circle if it exists
                if zoomed_circle is not None:
                    zoomed_circle.remove()

                # Draw a circle around the zoomed area
                zoomed_circle = plt.Circle((x, y), zoom_size + 10, fill=False, color='red', linestyle='dashed')
                ax_time_map.add_patch(zoomed_circle)

                # Add the canvas to the window
                canvas_zoom = FigureCanvasTkAgg(fig_zoom, master=zoom_window)
                canvas_zoom.draw()
                canvas_zoom.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                # Redraw the figure
                canvas.draw()


            else:
                print("No selection")

        def load_data():
            nonlocal gen_x_pixel, gen_y_pixel, gen_time, ax_time_map, highlight_points, cbar, counts

            if cbar is not None:
                cbar.remove()

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
                gen_x_map, gen_y_map = data[0].values.astype(float), data[1].values.astype(float)

                # Get the current choice from the dropdown menu
                choice = current_choice.get()
                # Clear the 3D plot
                ax_map.clear()

                # Calculate the pixel coordinates of the newly loaded data
                gen_x_pixel = np.floor((gen_x_map - x_min) / pixel_size).astype(int)
                gen_y_pixel = np.floor((gen_y_map - y_min) / pixel_size).astype(int)

                # Ensure that the pixel coordinates are within the valid range
                gen_x_pixel = np.clip(gen_x_pixel, 0, time_map.shape[1] - 1)
                gen_y_pixel = np.clip(gen_y_pixel, 0, time_map.shape[0] - 1)

                gen_time_and_weight = []
                for y, x in zip(gen_y_pixel, gen_x_pixel):
                    times = time_map[y, x, :time_counter[y, x]]
                    weights = weight_time_map[y, x, :time_counter[y, x]]
                    if not np.isnan(weights).all():
                        if choice == "show top 3 weight" and max(weights) >= 0.001:
                            gen_time_and_weight.append(
                                sorted(zip(times, weights), key=lambda tw: tw[1], reverse=True)[:3])
                        else:
                            gen_time_and_weight.append(list(zip(times, weights)))

                if gen_time_and_weight:
                    gen_time, gen_weight = zip(*[[list(tw) for tw in zip(*tws)] for tws in gen_time_and_weight])
                else:
                    gen_time, gen_weight = [], []

                # Convert the times and weights to strings and limit to 3 decimal places
                gen_time_str = [str([f"{t:.0f}" for t in times]) for times in gen_time]
                # Convert the weights to strings and limit to 3 decimal places
                gen_weight_str = [str([f"{w:.3f}" for w in weights]) for weights in gen_weight]

                position_weight = [weight_map[y, x] for y, x in zip(gen_y_pixel, gen_x_pixel)]

                # Convert the weights to strings and limit to 3 decimal places
                position_weight_str = [f"{w:.3f}" for w in position_weight]

                # Add the points to the time map heatmap
                points = ax_time_map.scatter(gen_x_pixel, gen_y_pixel, c='red', s=40)
                highlight_points.append(points)

                # Apply GMM to the pixel coordinates
                n_clusters = len(gen_x_pixel) // 100
                gmm = GaussianMixture(n_components=n_clusters)
                gmm.fit(np.vstack([gen_x_pixel, gen_y_pixel]).T)

                # Get the labels of the clusters
                labels = gmm.predict(np.vstack([gen_x_pixel, gen_y_pixel]).T)

                print('Estimated number of clusters: %d' % n_clusters)

                # Count the number of fitted points at each pixel
                counts = np.zeros((detector_height, detector_width))
                for x, y in zip(gen_x_pixel, gen_y_pixel):
                    counts[y, x] += 1

                # Clear the treeview widget
                cluster_treeview.delete(*cluster_treeview.get_children())

                # Clear the treeview widget
                treeview.delete(*treeview.get_children())

                # Iterate over the clusters
                for cluster_id in range(n_clusters):
                    # Get the points in the current cluster
                    points = np.vstack([gen_x_pixel[labels == cluster_id], gen_y_pixel[labels == cluster_id]]).T

                    # Compute the centroid of the cluster
                    centroid = np.mean(points, axis=0)

                    # Compute the furthest point in the cluster from its centroid
                    furthest_point = points[cdist([centroid], points).argmax()]

                    # Compute the direction vectors from the centroid to its furthest point
                    direction = furthest_point - centroid

                    print('Main direction of cluster %d: %s' % (cluster_id, direction))

                    # Compute the time and weight at the centroid
                    centroid_times = time_map[int(round(centroid[1])), int(round(centroid[0]))]
                    centroid_weights = weight_time_map[int(round(centroid[1])), int(round(centroid[0]))]

                    # Sort the times and weights by weight
                    centroid_times_and_weights = sorted(zip(centroid_times, centroid_weights), key=lambda tw: tw[1],
                                                        reverse=True)
                    if choice == "show top 3 weight":
                        centroid_times_and_weights = centroid_times_and_weights[:3]

                    # Separate the times and weights again
                    centroid_times, centroid_weights = zip(*centroid_times_and_weights)

                    centroid_time_str = str([f"{t:.0f}" for t in centroid_times])
                    centroid_weight_str = str([f"{w:.3f}" for w in centroid_weights])
                    centroid_position_weight_str = f"{weight_map[int(round(centroid[1])), int(round(centroid[0]))]:.3f}"

                    # Remove brackets and quotes from the string representation of the lists for display
                    centroid_times_display = centroid_time_str[1:-1].replace("'", "")
                    centroid_weight_display = centroid_weight_str[1:-1].replace("'", "")

                    # Add the cluster centroid, time, and weights to the cluster Treeview widget
                    cluster_treeview.insert('', 'end',
                                            values=(
                                                centroid[0], centroid[1], centroid_times_display,
                                                centroid_weight_display,
                                                centroid_position_weight_str))  # Add each item to the end of the Treeview

                    # Plot the direction vector on ax_map
                    ax_map.arrow(centroid[0], centroid[1], direction[0], direction[1],
                                 head_width=5, head_length=5, fc='red', ec='red')

                    # Get the times of the points in the current cluster
                    times = time_map[gen_y_pixel[labels == cluster_id], gen_x_pixel[labels == cluster_id]]

                    # Compute the projection of the points on the direction vector
                    projections = np.dot(points - centroid, direction)

                    # Sort the times based on the projections
                    sorted_times = times[np.argsort(projections)]

                    # Update the times of the points in the current cluster
                    time_map[gen_y_pixel[labels == cluster_id], gen_x_pixel[labels == cluster_id]] = sorted_times

                # Remove the old images
                if ax_map.images:
                    for im in ax_map.images:
                        im.remove()

                # Use PowerNorm instead of Normalize
                norm = PowerNorm(gamma=0.5, vmin=counts.min(), vmax=counts.max())

                # Create the image with the new norm
                im = ax_map.imshow(counts, cmap='viridis', norm=norm, origin='lower')

                # Add a colorbar
                cbar = plt.colorbar(im, ax=ax_map)
                cbar.set_label('N-times')

                ax_map.set_title('Detector Time Map')
                ax_map.set_xlabel('X')
                ax_map.set_ylabel('Y')

                canvas.draw()

                with open('map_result.dat', 'w') as f:
                    for x, y, times, weight, position in zip(gen_x_pixel, gen_y_pixel, gen_time_str, gen_weight_str,
                                                             position_weight_str):
                        # Remove quotes, then parse the weight string back into a list of floats
                        weights = [float(w.replace("'", "")) for w in weight[1:-1].split(", ")]

                        # check if all weights are less than 0.001
                        if all(w < 0.001 for w in weights):
                            # if so, skip this point
                            continue

                        # Remove brackets and quotes from the string representation of the lists for display
                        times_display = times[1:-1].replace("'", "")
                        weight_display = weight[1:-1].replace("'", "")

                        treeview.insert('', 'end',
                                        values=(
                                            x, y, times_display, weight_display,
                                            position))  # Add each item to the end of the Treeview
                        f.write(f"{x}, {y}, {times}, {weight}, {position}\n")  # Write the result to the file

        def create_stacked_area_plot(x_pixel, y_pixel):
            # Get the weights and times for this pixel
            weights = weight_time_map[y_pixel, x_pixel, :]
            times = time_map[y_pixel, x_pixel, :len(weights)]

            # Check if the pixel is on the function trajectory
            if np.all(weights == 0):
                # The pixel is not on the function trajectory
                messagebox.showinfo("Info", "The selected pixel is not on the function trajectory.")
                return

            # Check if the weights array contains valid values
            if np.isnan(weights).all() or np.isinf(weights).all() or len(weights) == 0:
                # The weights array does not contain valid values
                messagebox.showinfo("Info", "The selected pixel does not have valid weights.")
                return

            # Create the new window
            plot_window = tk.Toplevel()
            plot_window.title("Stacked Area Plot")
            plot_window.geometry("600x400")  # Set the initial size of the window

            # Add the pixel label
            pixel_label = tk.Label(master=plot_window, text=f"Pixel: ({x_pixel}, {y_pixel})")
            pixel_label.pack()

            # Create the new figure
            fig_map, ax_map = plt.subplots()

            # Create the stacked area plot
            ax_map.fill_between(times, weights, color='b', alpha=0.5)

            # Set the y-axis limit
            min_weight = weights.min() if not np.isnan(weights.min()) else 0
            max_weight = weights.max() if not np.isnan(weights.max()) else 1
            print(max_weight, min_weight)
            ax_map.set_ylim([min_weight - 0.05, max_weight + 0.05])  # Added the delta to the min and max weights

            # Add the canvas to the window
            canvas_map = FigureCanvasTkAgg(fig_map, master=plot_window)
            canvas_map.draw()
            canvas_map.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def on_pixel_click(event):
            # Check if the click is within the image bounds
            if event.xdata is None or event.ydata is None:
                return

            # Get the coordinates of the clicked pixel
            x_pixel = int(event.xdata)
            y_pixel = int(event.ydata)

            # Check if the clicked axes is one of the desired subplots
            if event.inaxes in [ax_weight_map, ax_time_map]:  # replace with your actual axes variables
                # Create the stacked area plot for this pixel
                create_stacked_area_plot(x_pixel, y_pixel)

            # Check if the clicked axes is the ax_map
            if event.inaxes == ax_map:
                # Create the enlarged view for this pixel
                count_map_zoom_window(x_pixel, y_pixel)

        def count_map_zoom_window(x_pixel, y_pixel):
            nonlocal plot_window

            # Remove old window
            if plot_window:
                plot_window.destroy()

            # Define the size of the area to be enlarged
            zoom_size = 100

            # Calculate the coordinates of the area to be enlarged
            x_start = max(0, x_pixel - zoom_size // 2)
            y_start = max(0, y_pixel - zoom_size // 2)
            x_end = min(detector_width, x_pixel + zoom_size // 2)
            y_end = min(detector_height, y_pixel + zoom_size // 2)

            # Extract the area from the counts
            enlarged_counts = counts[y_start:y_end, x_start:x_end]

            # Create the new window
            plot_window = tk.Toplevel()
            plot_window.title("Enlarged View")
            plot_window.geometry("600x400")  # Set the initial size of the window

            # Add the pixel label
            pixel_label = tk.Label(master=plot_window, text=f"Pixel: ({x_pixel}, {y_pixel})")
            pixel_label.pack()

            # Create the new figure
            fig_enlarged, ax_enlarged = plt.subplots()

            # Create the image
            im = ax_enlarged.imshow(enlarged_counts, cmap='viridis', interpolation='nearest', origin='lower')

            # Create a colorbar
            cbar = plt.colorbar(im, ax=ax_enlarged)
            cbar.set_label('N-times')

            ax_enlarged.set_title('Enlarged View')
            ax_enlarged.set_xlabel('X')
            ax_enlarged.set_ylabel('Y')

            # Add the canvas to the window
            canvas_enlarged = FigureCanvasTkAgg(fig_enlarged, master=plot_window)
            canvas_enlarged.draw()
            canvas_enlarged.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        counts, time_map, weight_map, weight_time_map, detector_width, detector_height, time_counter, x_min, y_min = function_analysis.calculate_map(
            fit_x, fit_y,
            pixel_size,
            sigma,
            delta_time,
            progress_callback)

        # Create a new window to display the image
        image_window = tk.Toplevel()
        image_window.title("Detector Time Map")
        image_window.geometry("800x800")  # Set the initial size of the window
        # Configure the image_window to distribute space between rows and columns
        image_window.grid_rowconfigure(0, weight=2)  # for the canvas row
        image_window.grid_rowconfigure(2, weight=3)  # for the treeview frame row
        image_window.grid_columnconfigure(0, weight=1)  # for the left part
        image_window.grid_columnconfigure(1, weight=1)  # for the right part

        # Create a new figure
        fig = plt.figure(figsize=(12, 4))

        # Add the 2D subplot
        ax_map = fig.add_subplot(131)  # Remove the projection='3d'

        # Plot the heatmap
        X, Y = np.meshgrid(np.arange(detector_width), np.arange(detector_height))
        # Use PowerNorm instead of Normalize
        norm = PowerNorm(gamma=0.5, vmin=counts.min(), vmax=counts.max())

        # Create the image with the new norm
        im = ax_map.imshow(counts, cmap='viridis', norm=norm, origin='lower')

        # Create a colorbar
        cbar = plt.colorbar(im, ax=ax_map)
        cbar.set_label('N-times')

        ax_map.set_title('Detector Time Map')
        ax_map.set_xlabel('X')
        ax_map.set_ylabel('Y')

        # Add the weight map heatmap
        ax_weight_map = fig.add_subplot(132)
        cax3 = ax_weight_map.imshow(weight_map, cmap='viridis', interpolation='nearest',
                                    origin='lower', norm=PowerNorm(0.2))  # use PowerNorm here
        fig.colorbar(cax3, ax=ax_weight_map)
        ax_weight_map.set_title('Weight Map')
        ax_weight_map.set_xlabel('Click on the pixel to see its time weight')

        # Add the time map heatmap
        ax_time_map = fig.add_subplot(133)
        first_times = time_map[:, :, 0]  # Extract the first time at each pixel
        cax2 = ax_time_map.imshow(first_times, cmap='viridis', interpolation='nearest', origin='lower')
        fig.colorbar(cax2, ax=ax_time_map)
        ax_time_map.set_title('Time Map')

        # Add the canvas to the window
        canvas = FigureCanvasTkAgg(fig, master=image_window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)

        # Bind the mouse click event to the on_pixel_click function
        canvas.mpl_connect('button_press_event', on_pixel_click)

        # Create the "Load Data" button
        load_button = tk.Button(master=image_window, text="Load Data", command=load_data, width=15, height=2)
        load_button.grid(row=1, column=0, sticky='w', padx=10, pady=10)

        # Create a new frame to hold the treeviews
        treeview_frame = tk.Frame(master=image_window)
        treeview_frame.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)

        # Configure the treeview frame to distribute space between treeviews
        treeview_frame.grid_rowconfigure(0, weight=1)  # for the label row
        treeview_frame.grid_rowconfigure(1, weight=5)  # for the treeview row
        treeview_frame.grid_columnconfigure(0, weight=1)
        treeview_frame.grid_columnconfigure(1, weight=1)

        # Create labels for the Treeviews
        treeview_label = tk.Label(master=treeview_frame, text="Time Values and Weights")
        treeview_label.grid(row=0, column=0, sticky='n', padx=10)
        cluster_treeview_label = tk.Label(master=treeview_frame, text="Cluster Point Values")
        cluster_treeview_label.grid(row=0, column=1, sticky='n', padx=10)

        # Create the Treeview widget for displaying time values and weights
        treeview = ttk.Treeview(master=treeview_frame, columns=("X", "Y", "Time", "Time Weight", "Position Weight"),
                                show="headings")
        treeview.bind('<<TreeviewSelect>>', on_listbox_select)
        treeview.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)

        # Create the Treeview widget for displaying cluster point values
        cluster_treeview = ttk.Treeview(master=treeview_frame, columns=(
            "Cluster X", "Cluster Y", "Cluster Time", "Cluster Time Weight", "Cluster Position Weight"),
                                        show="headings")
        cluster_treeview.bind('<<TreeviewSelect>>', on_cluster_listbox_select)
        cluster_treeview.grid(row=1, column=1, sticky='nsew', padx=10, pady=10)

        # Create the variable to store the current choice
        current_choice = tk.StringVar()

        # Set the default value
        current_choice.set("show all time")

        # Create the dropdown menu
        dropdown = tk.OptionMenu(image_window, current_choice, "show all time", "show top 3 weight")
        dropdown.grid(row=1, column=1, sticky='w', padx=10, pady=10)

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

            # Dynamically add A and p to the params dictionary for each component of gen_x
            for i, (Ax, px) in enumerate(zip(A_x, p_x), 1):
                params[f'A_x{i}'] = Ax
                params[f'p_x{i}'] = px

            # Dynamically add A and p to the params dictionary for each component of gen_y
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

            fig_load, axs = plt.subplots(2, 1, figsize=(6, 6))

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

            canvas_plot = FigureCanvasTkAgg(fig_load, master=plot_frame)
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

    def run_fit_in_thread(x, y, params, filter_press_count, progress_var, progress_window, status_label, pixel_size,
                          sigma, delta_time):

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
                draw_plot(ax, canvas, gen_x, gen_y, 'Original and Fitted Data', 'Original data', clear=False,
                          scatter=True)
                draw_plot(ax, canvas, fit_x2, fit_y2, 'New Data', 'Filtered data', clear=False, scatter=False)
                # 标记每过50个点的节点
                for i in range(0, len(fit_x2), 50):
                    label = f'({fit_x2[i]:.2f}, {fit_y2[i]:.2f}, {i})'
                    ax.annotate(label, (fit_x2[i], fit_y2[i]), xytext=(5, -10),
                                textcoords='offset points', ha='left', va='top')
                    ax.plot(fit_x2[i], fit_y2[i], 'ro', markersize=5)  # Highlighted points are red circles

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

        progress_var = tk.DoubleVar()
        progress_var.set(0)

        status_label = tk.Label(progress_window, text="Initializing...", font=("Arial", 12))
        status_label.pack()

        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=300)
        progress_bar.pack(padx=20, pady=20)

        # Start a new thread to run the time-consuming fitting operation
        fit_thread = threading.Thread(target=run_fit_in_thread,
                                      args=(gen_x, gen_y, params, filter_press_count, progress_var, progress_window,
                                            status_label, pixel_size, sigma, delta_time))
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
        sigma_var = tk.StringVar(value="1")  # Default value
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
    zoom_window = None
    plot_window = None

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

    auto_scale_var = tk.IntVar()

    checks_frame = ttk.Frame(app)
    checks_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

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
