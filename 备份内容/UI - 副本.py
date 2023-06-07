import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Generate import run_gen
from Filter import run_odr
import math


def safe_eval(expr):
    allowed = {
        'math': math,
        'pi': math.pi,
    }
    names = set(name for name in expr.split() if name.isalpha())

    # Ensure that all variables in an expression are whitelisted
    for name in names:
        if name not in allowed:
            raise ValueError(f'Not allowed: {name}')

    # Calculate the value of an expression
    return eval(expr, {'__builtins__': {}}, allowed)


def create_app():
    # 创建Tkinter应用窗口
    app = tk.Tk()
    app.title('Data Generator and ODR')

    # 创建用于输入参数的输入框，每个输入框都有一个预设值
    tk.Label(app, text="T").grid(row=0)
    tk.Label(app, text="A").grid(row=1)
    tk.Label(app, text="w").grid(row=2)
    tk.Label(app, text="p1").grid(row=3)
    tk.Label(app, text="p2").grid(row=4)
    tk.Label(app, text="n").grid(row=5)

    entries = []
    presets = ['1', '1', '1', '0', 'pi', '100']  # 预设值
    for i in range(6):
        entry = tk.Entry(app)
        entry.insert(0, presets[i])  # 为输入框添加预设值
        entry.grid(row=i, column=1, padx=10, pady=10)  # 设置间距
        entries.append(entry)

    # 创建用于显示matplotlib图像的画布
    fig = Figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=app)
    canvas.get_tk_widget().grid(row=0, column=2, rowspan=6, padx=10, pady=10)  # 设置间距

    # 创建按钮，用于生成图像和运行ODR
    def generate():
        # 从输入框中获取参数，并尝试将它们转换为浮点数
        T = safe_eval(entries[0].get())
        A = safe_eval(entries[1].get())
        w = safe_eval(entries[2].get())
        p1 = safe_eval(entries[3].get())
        p2 = safe_eval(entries[4].get())
        n = int(entries[5].get())

        # 使用参数生成数据并绘图
        x, y = run_gen(T, A, w, p1, p2, n)
        ax.clear()
        ax.scatter(x, y, label='Output')
        ax.set_title('Output')
        canvas.draw()

    def filter():
        # 运行ODR，并使用结果重新绘图
        x_fit, y_fit = run_odr()
        ax.clear()
        ax.plot(x_fit, y_fit, label='ODR fit')
        ax.set_title('ODR fit to data')
        canvas.draw()

    tk.Button(app, text="Generate", command=generate).grid(row=6, column=0, padx=10, pady=10)  # 设置间距
    tk.Button(app, text="Filter", command=filter).grid(row=6, column=1, padx=10, pady=10)  # 设置间距

    # 运行应用
    app.mainloop()


if __name__ == '__main__':
    create_app()
