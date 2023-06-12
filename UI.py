import tkinter as tk
import tkinter.messagebox
from tkinter import ttk
from ttkbootstrap import Style
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
#from matplotlib.backend_bases import MouseEvent
from Generate import run_gen
from Filter import run_fit
import math

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

    entries = [
        create_label_and_entry(app, parameter, i) for i, parameter in enumerate(PARAMETERS)
    ]

    fig = Figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111)

    canvas = FigureCanvasTkAgg(fig, master=app)
    canvas.get_tk_widget().grid(row=0, column=2, rowspan=8, padx=5, pady=5)

    # Initialize cursor as None
    cursor = None

    # Initialize generate result as None
    gen_x, gen_y = None, None
    T_uniform = None

    # Create a cursor class to show the x,y position
    class Cursor(object):
        def __init__(self, ax):
            self.ax = ax
            self.lx = ax.axhline(color='b')  # the horizontal line
            self.ly = ax.axvline(color='b')  # the vertical line

            # text location in axes coordinates
            self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

        def mouse_move(self, event):
            if not event.inaxes:
                return

            x, y = event.xdata, event.ydata
            # update the line positions
            self.lx.set_ydata([y])  # use a list
            self.ly.set_xdata([x])  # use a list

            self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()

    def generate():
        nonlocal cursor, gen_x, gen_y, T_uniform
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
            ax.clear()
            ax.scatter(x, y, label='Output')
            ax.set_title('Output')
            canvas.draw()
            gen_x, gen_y = x, y  # store generated values
            T_uniform = np.linspace(0, T, n)
            cursor = Cursor(ax)
            canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e))

    def filter():
        nonlocal cursor, gen_x, gen_y, T_uniform
        try:
            if gen_x is not None and gen_y is not None:  # check if data is already generated
                ax.scatter(gen_x, gen_y, label='Original data')  # draw generated data again
            fit_results = run_fit()  # run_fit now returns a dict
            fit_x, fit_y = fit_results["fit_x"], fit_results["fit_y"]

            fit_x2 = fit_results["A"] * np.cos(fit_results["w1"] * np.pi * T_uniform + fit_results["p1"])
            fit_y2 = fit_results["B"] * np.cos(fit_results["w2"] * np.pi * T_uniform + fit_results["p2"])
            # x = A * np.cos(w1 * np.pi * t + p1) + np.random.normal(0, 0.02, n)  # mean,variance,count
            # y = B * np.cos(w2 * np.pi * t + p2) + np.random.normal(0, 0.02, n)




            ax.clear()
            ax.scatter(gen_x, gen_y, label='Original data')
            #ax.scatter(fit_x, fit_y, color='red', label='Filtered data')  # change to scatter for fitted points
            ax.plot(fit_x2, fit_y2, color='red', label='Filtered data')
            ax.set_title('Original and Filtered Data')
            ax.legend()
            canvas.draw()
            cursor = Cursor(ax)
            canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e))

        # Display the fitted parameters
        for param, value in fit_results.items():
            if param != "fit_x" and param != "fit_y":
                print(f"{param}: {value}")

    # Create a style for the button
    generate_button = ttk.Button(app, text="Generate", command=generate, style="success.TButton")
    generate_button.grid(row=8, column=0, padx=10, pady=10)

    filter_button = ttk.Button(app, text="Filter", command=filter, style="info.TButton")
    filter_button.grid(row=8, column=1, padx=10, pady=10)

    app.mainloop()


if __name__ == '__main__':
    create_app()
