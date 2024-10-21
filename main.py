import tkinter as tk
from gui.main_gui import BorohydridePredictionTool  # 导入你的GUI类

def main():
    # 创建主窗口
    root = tk.Tk()

    # 实例化GUI应用
    app = BorohydridePredictionTool(root)

    # 运行Tkinter主循环
    root.mainloop()

if __name__ == "__main__":
    main()
