import math
import tkinter as tk
import tkinter.messagebox as messagebox
from tkinter import ttk, filedialog, Toplevel, Label, Scale
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ttkbootstrap import Style
from Filter import run_fit
from Generate import run_gen
from tkinter import scrolledtext
import json
import pandas as pd


PARAMETERS = [
    {"label": "T", "preset": '1', "tooltip": 'Input range: Any positive number'},
    {"label": "A", "preset": '1', "tooltip": 'Input range: Any positive number'},
    {"label": "B", "preset": '1', "tooltip": 'Input range: Any positive number'},
    {"label": "w1", "preset": '1', "tooltip": 'Input range: Any positive number'},
    {"label": "w2", "preset": '1', "tooltip": 'Input range: Any positive number'},
    {"label": "p1", "preset": '0', "tooltip": 'Input range: x*pi or pi/x'},
    {"label": "p2", "preset": 'pi', "tooltip": 'Input range: x*pi or pi/x'},
    {"label": "n", "preset": '100', "tooltip": 'Input range: Better Larger than 0'}
]


class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def create_tool_tip(widget, text):
    tool_tip = ToolTip(widget)

    def enter(event):
        tool_tip.showtip(text)

    def leave(event):
        tool_tip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


def safe_eval(expr):
    allowed = {
        'math': math,
        'pi': math.pi,
    }
    names = set(name for name in expr.split() if name.isalpha())

    for name in names:
        if name not in allowed:
            raise ValueError(f'Not allowed: {name}')

    return eval(expr, {'__builtins__': {}}, allowed)


def create_label_and_entry(app, parameter, row):
    label_text = parameter["label"]
    preset_value = parameter["preset"]
    tooltip_text = parameter["tooltip"]
    label = ttk.Label(app, text=label_text, font=('Arial', 10))
    label.grid(row=row, column=0, padx=5, pady=5)
    entry = ttk.Entry(app, font=('Arial', 10))
    entry.insert(0, preset_value)
    entry.grid(row=row, column=1, padx=5, pady=5)
    create_tool_tip(entry, tooltip_text)
    return entry


def create_app():
    app = tk.Tk()
    app.style = Style(theme='flatly')

    # Configure row and column weights
    for i in range(12):
        app.rowconfigure(i, weight=1)
    for i in range(3):
        app.columnconfigure(i, weight=1)

    entries = [
        create_label_and_entry(app, parameter, i) for i, parameter in enumerate(PARAMETERS)
    ]

    fig = Figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111)

    # Create the canvas as a child of a Frame
    frame = tk.Frame(app)
    frame.grid(row=0, column=2, rowspan=8, padx=5, pady=5, sticky='nsew')
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill='both', expand=True)


    #checkbuttons
    check_var1 = tk.IntVar(value=1)  # set initial to true
    check_var2 = tk.IntVar(value=1)

    check_button1 = ttk.Checkbutton(app, text="Show Original Data", variable=check_var1, style="TCheckbutton")
    check_button1.grid(row=9, column=0, padx=10, pady=10)

    check_button2 = ttk.Checkbutton(app, text="Show Fitted Data", variable=check_var2, style="TCheckbutton")
    check_button2.grid(row=9, column=1, padx=10, pady=10)

    data_loaded = False
    parameters_loaded = False
    filter_press_count = 0  # initialize counter at global scope



    # 创建一个字典来存储beta_limit的值
    beta_limit_dict = {"A": 0.05, "B": 0.05, "w1": 0.05, "w2": 0.05, "p1": 0.05, "p2": 0.05}

    # Sub-window
    def open_limit_window():
        limit_window = Toplevel(app)
        limit_window.title("Set Beta Limit")

        # 创建一个函数来更新beta_limit字典的值
        def update_beta_limit(param, val):
            beta_limit_dict[param] = round(float(val), 4)
            # 更新标签显示的文本
            beta_limit_labels[param].config(text=f"Beta Limit {param}: {beta_limit_dict[param]}")

        # 为每个参数创建一个滑动条和标签
        beta_limit_labels = {}
        for i, param in enumerate(["A", "B", "w1", "w2", "p1", "p2"]):
            beta_limit_labels[param] = Label(limit_window, text=f"Beta Limit {param}: {beta_limit_dict[param]}")
            beta_limit_labels[param].grid(row=i, column=1, padx=3, pady=10, sticky='w')

            scale = Scale(limit_window, from_=0, to=0.5, resolution=0.01, orient='horizontal',
                          command=lambda val, p=param: update_beta_limit(p, val))
            scale.grid(row=i, column=0, padx=(0, 5), pady=(0, 10), sticky='e')
            scale.set(0.05)  # 设置初始值为5%
        # 在主窗口创建一个按钮

    button_set_limit = ttk.Button(app, text="Set Limit", command=open_limit_window)
    button_set_limit.grid(row=11, column=0, padx=(0, 5), pady=(0, 10), sticky='e')

    def show_logs():

        log_window = tk.Toplevel(app)

        txt = scrolledtext.ScrolledText(log_window, undo=True)
        txt['font'] = ('consolas', '12')
        txt.pack(expand=True, fill='both')

        # Load log file
        with open('filter_log.txt', 'r') as log_file:
            log_contents = log_file.read()

        # Insert log contents to the text widget
        txt.insert(tk.INSERT, log_contents)

    # Create a 'Log' button
    log_button = tk.Button(app, text="Log", command=show_logs)
    log_button.grid(row=0, column=2, padx=(0, 5), pady=(0, 10), sticky='e')

    # status_label = ttk.Label(app, text="Status: Data not loaded, Parameters not loaded", font=('Arial', 10))
    # status_label.grid(row=10, column=0, columnspan=4, padx=5, pady=5)

    status_frame = ttk.Frame(app)
    status_frame.grid(row=11, column=2, columnspan=2, padx=5, pady=0)

    data_status_label = ttk.Label(status_frame, text="Data: Not Loaded", background="red", font=('Arial', 10))
    data_status_label.pack(side="left", fill="x", expand=True)

    parameters_status_label = ttk.Label(status_frame, text="Parameters: Not Loaded", background="red", font=('Arial', 10))
    parameters_status_label.pack(side="left", fill="x", expand=True)


    def draw_plot(ax, canvas, x, y, title, label, clear=True, scatter=True):
        if clear:
            ax.clear()
        if scatter:
            ax.scatter(x, y, label=label)
        else:
            ax.plot(x, y, color='red', label=label)
        ax.set_title(title)
        ax.legend()
        canvas.draw()

    def load_data():
        nonlocal gen_x, gen_y, T_uniform, data_loaded
        file_path = filedialog.askopenfilename(filetypes=[('Data Files', '*.dat'), ('Data Files', '*.json')])
        if file_path:
            if file_path.endswith('.dat'):
                df = pd.read_csv(file_path, header=None)
                gen_x, gen_y = df[0].values, df[1].values

            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data_list = data["Generated Data"]
                    gen_x, gen_y = np.array([item[0] for item in data_list]), np.array([item[1] for item in data_list])

            n = len(gen_x)
            T = safe_eval(entries[0].get())
            T_uniform = np.linspace(0, T, n)
            data_loaded = True

            # New code to plot the data right after loading
            ax.clear()
            ax.scatter(gen_x, gen_y, label='Loaded data')
            ax.set_title('Loaded Data')
            ax.legend()
            canvas.draw()

            update_status_label()
        if data_loaded:
            data_status_label.config(text="Data: Loaded", background="green")

    def load_parameters():
        nonlocal params
        file_path = filedialog.askopenfilename(filetypes=[('Parameter Files', '*.json'), ('Parameter Files', '*.dat')])
        if file_path:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    parameters = json.load(f)
                    params = parameters
                    for i, entry in enumerate(entries):
                        entry.delete(0, 'end')
                        entry.insert(0, str(parameters[PARAMETERS[i]['label']]))

            elif file_path.endswith('.dat'):
                df = pd.read_csv(file_path, header=None)
                parameters = df.to_dict(orient='records')[0]
                params = parameters
                for i, entry in enumerate(entries):
                    entry.delete(0, 'end')
                    entry.insert(0, str(parameters[PARAMETERS[i]['label']]))

            nonlocal T_uniform
            n = int(entries[7].get())
            T = safe_eval(entries[0].get())
            T_uniform = np.linspace(0, T, n)

        nonlocal parameters_loaded
        parameters_loaded = True
        update_status_label()
        if parameters_loaded:
            parameters_status_label.config(text="Parameters: Loaded", background="green")

    def update_status_label():
        if data_loaded and parameters_loaded:
            filter_button['state'] = 'normal'


    # Checkbutton state change callback
    def redraw_on_check():
        filter()

    # Attach the callback to the Checkbuttons
    check_var1.trace('w', lambda *args: redraw_on_check())
    check_var2.trace('w', lambda *args: redraw_on_check())

    # Initialize
    cursor = None
    gen_x, gen_y = None, None
    T_uniform = None
    params = None

    # Create a cursor class to show the x,y position
    class Cursor(object):
        def __init__(self, ax):
            self.ax = ax
            self.lx = ax.axhline(color='b')
            self.ly = ax.axvline(color='b')
            self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

        def mouse_move(self, event):
            if not event.inaxes:
                return

            x, y = event.xdata, event.ydata
            # update the line positions
            self.lx.set_ydata([y])
            self.ly.set_xdata([x])

            self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()

    def generate():
        nonlocal cursor, gen_x, gen_y, T_uniform, params
        gen_x, gen_y, T_uniform, params = None, None, None, None # To fix the bug mentioned 15/6/2023, just set params to None
        try:
            T = safe_eval(entries[0].get())
            A = safe_eval(entries[1].get())
            B = safe_eval(entries[2].get())
            w1 = safe_eval(entries[3].get())
            w2 = safe_eval(entries[4].get())
            p1 = safe_eval(entries[5].get())
            p2 = safe_eval(entries[6].get())
            n = int(entries[7].get())

            x, y = run_gen(T, A, B, w1, w2, p1, p2, n)
            draw_plot(ax, canvas, x, y, 'Output', 'Output', clear=True, scatter=True)
            gen_x, gen_y = x, y  # store generated values
            T_uniform = np.linspace(0, T, n)
            cursor = Cursor(ax)
            canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
            filter_button['state'] = 'normal'
            # store generated parameters
            params = {
                'T': T,
                'A': A,
                'B': B,
                'w1': w1,
                'w2': w2,
                'p1': p1,
                'p2': p2,
                'n': n,
            }

        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e))

    def filter():
        nonlocal cursor, gen_x, gen_y, T_uniform, params, beta_limit_dict, filter_press_count

        try:
            filter_press_count += 1  # increment counter
            fit_results = run_fit(gen_x, gen_y, params, beta_limit_dict, filter_press_count)

            fit_x2 = fit_results["A"] * np.cos(fit_results["w1"] * np.pi * T_uniform + fit_results["p1"])
            fit_y2 = fit_results["B"] * np.cos(fit_results["w2"] * np.pi * T_uniform + fit_results["p2"])

            ax.clear()
            if check_var1.get() == 1:
                draw_plot(ax, canvas, gen_x, gen_y, 'Original and Fitted Data', 'Original data', clear=False,
                          scatter=True)
            if check_var2.get() == 1:
                draw_plot(ax, canvas, fit_x2, fit_y2, 'Original and Fitted Data', 'Filtered data', clear=False,
                          scatter=False)
            cursor = Cursor(ax)
            canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e))
        print(beta_limit_dict)

        # Create a new Toplevel window to display the parameters
        new_window = tk.Toplevel(app)

        # Create a Treeview widget
        tree = ttk.Treeview(new_window, columns=('Parameters', 'Values'), show='headings')
        tree.heading('Parameters', text='Parameters', anchor=tk.CENTER)
        tree.heading('Values', text='Values', anchor=tk.CENTER)

        # Change the column width and alignment
        tree.column('Parameters', width=100, anchor=tk.CENTER)
        tree.column('Values', width=500, anchor=tk.CENTER)

        # Insert the parameter limits
        #tree.insert('', 'end', values=("Parameter limits", str(beta_limit_dict)))

        # Insert the parameter values
        for param, value in fit_results.items():
            if param != "fit_x" and param != "fit_y":
                tree.insert('', 'end', values=(param, value))

        tree.pack()

    buttons_frame = ttk.Frame(app)
    buttons_frame.grid(row=8, column=0, columnspan=4, padx=10, pady=10, sticky='nsew')
    for i in range(4):
        buttons_frame.grid_columnconfigure(i, weight=1)

    generate_button = ttk.Button(buttons_frame, text="Generate", command=generate, style="info.TButton")
    generate_button.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

    # Here, we use buttons_frame as the parent widget instead of app
    filter_button = ttk.Button(buttons_frame, text="Fit", command=filter, style="info.TButton", state='disabled')
    filter_button.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

    load_data_button = ttk.Button(buttons_frame, text="Load Data", command=load_data, style="info.TButton")
    load_data_button.grid(row=0, column=2, padx=10, pady=10, sticky='nsew')

    load_parameters_button = ttk.Button(buttons_frame, text="Load Parameters", command=load_parameters,
                                        style="info.TButton")
    load_parameters_button.grid(row=0, column=3, padx=10, pady=10, sticky='nsew')

    app.mainloop()


if __name__ == '__main__':
    create_app()
