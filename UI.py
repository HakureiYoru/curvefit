import tkinter
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk, filedialog, Toplevel, Label, Scale
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ttkbootstrap import Style
from Filter import run_fit
from tkinter import scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
import function_analysis
import threading
import time


def create_app():
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

            f1, f2, _, _, _, _, _, _ = [float(val[1:]) if isinstance(val, str) and val.startswith('#') else val for val
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
            f_x = analysis_results["gen_x_frequencies"]
            f_y = analysis_results["gen_y_frequencies"]
            # print("--------------")
            # print(f"gen_x_amplitudes: {A_x}")
            # print(f"gen_x_phases: {p_x}")
            # print(f"gen_y_amplitudes: {B_y}")
            # print(f"gen_y_phases: {p_y}")
            # print(f"gen_x_frequencies: {f_x}")
            # print(f"gen_y_frequencies: {f_y}")
            # print("--------------")

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

    def run_fit_in_thread(x, y, params, filter_press_count, progress_var, progress_window, status_label):
        try:
            def progress_callback(xk, convergence, progress_range=(0, 50)):
                # 我们将进度条划分为两个部分，这个回调函数用于更新前半部分
                current_progress = progress_var.get()
                progress_increment = (progress_range[1] - progress_range[0]) / 50  # 假设我们有50步
                progress_var.set(min(current_progress + progress_increment, progress_range[1]))

            # 设置标签文本为拟合过程
            status_label.config(text="Fitting in progress...")

            fit_results = run_fit(x, y, params, bounds_factor_dict, filter_press_count,
                                  progress_callback=progress_callback)
            fitted_params = fit_results["fitted_params"]

            # 设置标签文本为计算时间过程
            status_label.config(text="Calculating times...")

            estimated_times = fit_results["time_fit"]
            time_diffs = np.diff(estimated_times)

            # 模拟时间计算过程中的进度更新
            for i in range(50, 101, 10):
                time.sleep(0.5)  # 假设每步需要一些时间
                progress_var.set(i)

            # 设置标签文本为完成状态
            status_label.config(text="Completed")

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

                progress_window.destroy()

                # Create a new window to display the estimated times and their differences
                time_window = tk.Toplevel()
                time_window.title("Estimated Times and Time Differences")

                fig, axs = plt.subplots(2)
                axs[0].plot(estimated_times)
                axs[0].set_title('Estimated Times')
                axs[0].set_xlabel('Index')
                axs[0].set_ylabel('Time')

                axs[1].plot(time_diffs)
                axs[1].set_title('Time Differences')
                axs[1].set_xlabel('Index')
                axs[1].set_ylabel('Difference')

                canvas1 = FigureCanvasTkAgg(fig, master=time_window)
                canvas1.draw()
                canvas1.get_tk_widget().pack(expand=True, fill=tk.BOTH)

            app.after(0, update_ui)

        except Exception as e:
            tk.messagebox.showerror("Error", str(e))

    def Fit():
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
                                            status_label))
        fit_thread.daemon = True  # Set as a daemon thread so that when the main program exits the thread will also exit
        fit_thread.start()

    # Initialize
    data_loaded = False
    parameters_loaded = False
    filter_press_count = 0  # initialize counter at global scope
    previous_window = None
    gen_x, gen_y, t_measured = None, None, None
    params = None
    new_window = None
    param_window = None
    # Initial value 0.05
    # 定义空词典
    bounds_factor_dict = dict()

    app = tk.Tk()

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

    filter_button = ttk.Button(buttons_frame, text="Fit", command=Fit, state='disabled')
    filter_button.pack(side='top', padx=10, pady=10, fill='x')

    load_button = ttk.Button(buttons_frame, text="Load Data & Parameters", command=load_file)
    load_button.pack(side='top', padx=10, pady=10, fill='x')

    app.mainloop()


if __name__ == '__main__':
    create_app()
