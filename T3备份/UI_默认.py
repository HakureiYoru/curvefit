import tkinter as tk
import tkinter.messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from T3 import run_gen, gen
from T3_filter import run_fit
import math


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


def create_label_and_entry(app, text, preset, row, tooltip_text):
    label = tk.Label(app, text=text, font=('Arial', 10))
    label.grid(row=row, padx=5, pady=5)
    entry = tk.Entry(app, font=('Arial', 10))
    entry.insert(0, preset)
    entry.grid(row=row, column=1, padx=5, pady=5)
    create_tool_tip(entry, tooltip_text)
    return entry


def create_app():
    app = tk.Tk()
    app.title('Data Generator and ODR')
    app.configure(bg='white')

    labels = ["T", "A", "B", "w1", "w2", "p1", "p2", "n"]
    presets = ['1', '1', '1', '1', '1', '0', 'pi', '100']
    tooltips = [
        'Input range: Any positive number',
        'Input range: Any positive number',
        'Input range: Any positive number',
        'Input range: Any positive number',
        'Input range: Any positive number',
        'Input range: x*pi or pi/x',
        'Input range: x*pi or pi/x',
        'Input range: Better Larger than 0',
    ]

    entries = [
        create_label_and_entry(app, labels[i], presets[i], i, tooltips[i]) for i in range(8)
    ]

    fig = Figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=app)
    canvas.get_tk_widget().grid(row=0, column=2, rowspan=8, padx=5, pady=5)

    def generate():
        try:
            T = safe_eval(entries[0].get())
            A = safe_eval(entries[1].get())
            B = safe_eval(entries[2].get())
            w1 = safe_eval(entries[3].get())
            w2 = safe_eval(entries[4].get())
            p1 = safe_eval(entries[5].get())
            p2 = safe_eval(entries[6].get())
            n = int(entries[7].get())

            t, x, y = run_gen(T, A, B, w1, w2, p1, p2, n)
            ax.clear()
            ax.scatter(x, y, label='Output')
            ax.set_title('Output')
            canvas.draw()
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e))

    def filter():
        try:
            x_fit, y_fit = run_fit()
            t = np.linspace(0, 1, len(x_fit))  # 生成与x_fit长度相同的时间序列

            ax.plot(x_fit, y_fit, 'r-', label='curve_fit result')
            ax.set_title('ODR fit to data')
            ax.legend()
            canvas.draw()
        except Exception as e:
            tkinter.messagebox.showerror("Error", str(e))

    generate_button = tk.Button(app, text="Generate", command=generate, font=('Arial', 10))
    generate_button.grid(row=8, column=0, padx=10, pady=10)

    filter_button = tk.Button(app, text="Filter", command=filter, font=('Arial', 10))
    filter_button.grid(row=8, column=1, padx=10, pady=10)

    app.mainloop()


if __name__ == '__main__':
    create_app()
