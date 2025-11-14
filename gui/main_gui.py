import pandas as pd
import tkinter as tk
import os
from tkinter import filedialog, messagebox,ttk

from models import train, predictor
from features.feature_extraction import extract_features, extract_target

class BorohydridePredictionTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Borohydride Hydrogen Storage Property Prediction Tool")
        self.root.geometry("860x520")
        self.root.configure(bg="#e8ecf5")
        self.root.resizable(False, False)

        # 样式与容器
        button_base = {
            "font": ("Segoe UI", 11, "bold"),
            "width": 14,
            "padx": 8,
            "pady": 10,
            "relief": "flat",
            "cursor": "hand2"
        }

        container = tk.Frame(self.root, bg="#ffffff", bd=0, highlightthickness=0)
        container.place(relx=0.5, rely=0.5, anchor="center", width=760, height=420)

        header = tk.Frame(container, bg="#ffffff")
        header.pack(fill="x", padx=30, pady=(25, 15))

        title_label = tk.Label(
            header,
            text="Borohydride Property Prediction Tool",
            font=("Segoe UI", 20, "bold"),
            bg="#ffffff",
            fg="#1f3c88"
        )
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            header,
            text="Upload data, choose a model, and generate hydrogen bond predictions.",
            font=("Segoe UI", 11),
            bg="#ffffff",
            fg="#6e7a96"
        )
        subtitle_label.pack(anchor="w", pady=(6, 0))

        controls = tk.Frame(container, bg="#ffffff")
        controls.pack(fill="x", padx=30, pady=(0, 20))
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(3, weight=1)

        model_label = tk.Label(
            controls,
            text="Select Model",
            font=("Segoe UI", 11, "bold"),
            bg="#ffffff",
            fg="#53618a"
        )
        model_label.grid(row=0, column=0, sticky="w")

        self.model_var = tk.StringVar(value="Linear")
        model_menu = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=["Linear", "SVM", "Decision_tree", "Random_forest", "Lasso", "Ridge"],
            font=("Segoe UI", 11),
            state="readonly"
        )
        model_menu.grid(row=0, column=1, sticky="we", padx=(10, 20), ipady=3)
        model_menu.current(0)

        property_label = tk.Label(
            controls,
            text="Select Property",
            font=("Segoe UI", 11, "bold"),
            bg="#ffffff",
            fg="#53618a"
        )
        property_label.grid(row=0, column=2, sticky="w")

        self.property_var = tk.StringVar(value="Hydrogen Bond Length")
        property_menu = ttk.Combobox(
            controls,
            textvariable=self.property_var,
            values=["Hydrogen Bond Length"],
            font=("Segoe UI", 11),
            state="readonly"
        )
        property_menu.grid(row=0, column=3, sticky="we", padx=(10, 0), ipady=3)

        actions = tk.Frame(container, bg="#ffffff")
        actions.pack(fill="x", padx=30, pady=(0, 20))

        import_btn = tk.Button(
            actions,
            text="Import Data",
            command=self.import_data,
            bg="#38b000",
            fg="#ffffff",
            activebackground="#2d8c00",
            activeforeground="#ffffff",
            **button_base
        )
        import_btn.pack(side="left", padx=(0, 15))

        predict_btn = tk.Button(
            actions,
            text="Run Prediction",
            command=self.run_prediction,
            bg="#0078ff",
            fg="#ffffff",
            activebackground="#005fcc",
            activeforeground="#ffffff",
            **button_base
        )
        predict_btn.pack(side="left", padx=15)

        export_btn = tk.Button(
            actions,
            text="Export Results",
            command=self.export_results,
            bg="#ff6f3c",
            fg="#ffffff",
            activebackground="#e85a26",
            activeforeground="#ffffff",
            **button_base
        )
        export_btn.pack(side="left", padx=15)

        results_frame = tk.Frame(container, bg="#ffffff")
        results_frame.pack(fill="both", expand=True, padx=30, pady=(10, 25))
        results_frame.grid_rowconfigure(1, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

        results_label = tk.Label(
            results_frame,
            text="Predictions",
            font=("Segoe UI", 12, "bold"),
            bg="#ffffff",
            fg="#1f3c88"
        )
        results_label.grid(row=0, column=0, sticky="w", pady=(0, 8))

        tree_style = ttk.Style()
        tree_style.configure(
            "Prediction.Treeview",
            font=("Segoe UI", 10),
            rowheight=28,
            background="#f7f9fd",
            fieldbackground="#f7f9fd"
        )
        tree_style.configure("Prediction.Treeview.Heading", font=("Segoe UI", 11, "bold"))

        columns = ("formula", "predicted", "spacegroup")
        self.result_tree = ttk.Treeview(
            results_frame,
            columns=columns,
            show="headings",
            selectmode="none",
            style="Prediction.Treeview"
        )
        self.result_tree.heading("formula", text="formula")
        self.result_tree.heading("predicted", text="Predicted H_length")
        self.result_tree.heading("spacegroup", text="spacegroup.number")
        self.result_tree.column("formula", width=200, anchor="w")
        self.result_tree.column("predicted", width=180, anchor="center")
        self.result_tree.column("spacegroup", width=160, anchor="center")
        self.result_tree.grid(row=1, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(results_frame, command=self.result_tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        self.result_tree.insert("", "end", values=("No data", "Run prediction to see results", ""))

        self.data = None
        self.model = None
        self.predictions = None

    def import_data(self):
        # 打开文件对话框选择CSV文件
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

        if file_path:
            try:
                self.data = self.load_data(file_path)
                messagebox.showinfo("Success", "Data imported successfully!")
            except ValueError as e:
                messagebox.showwarning("Error", str(e))
        else:
            messagebox.showwarning("Error", "No file selected")

    def load_data(self, file_path):
        # 加载CSV数据，并提取特定列
        data = pd.read_csv(file_path)


        # 检查文件是否包含所需的特征
        required_columns = ['H_and_X_ar', 'electronegativity_Pauling_pow2']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # 只提取氢键长度和电负性列
        return data

    def run_prediction(self):
        # 检查是否已经导入数据
        if self.data is None:
            messagebox.showwarning("Error", "Please import data first")
            return

        train_data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'train_data.csv')
        )
        if not os.path.exists(train_data_path):
            messagebox.showwarning("Error", "Training data file not found")
            return

        data_train = pd.read_csv(train_data_path)
        features_train = extract_features(data_train)
        target = extract_target(data_train)

        # 检查是否选择了模型
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showwarning("Error", "Please choose a model")
            return

        # 根据选择的模型进行预测
        if model_name == "Linear":
            model = train.linear(features_train, target)
        elif model_name == "SVM":
            model = train.svm(features_train, target)
        elif model_name == "Decision_tree":
            model = train.decision_tree(features_train, target)
        elif model_name == "Random_forest":
            model = train.random_forest(features_train, target)
        elif model_name == "Lasso":
            model = train.lasso(features_train, target)
        elif model_name == "Ridge":
            model = train.ridge(features_train, target)

        # 使用训练好的模型进行预测
        features = extract_features(self.data)
        self.predictions = model.predict(features)

        # 添加预测结果到数据中
        self.data['Predicted H_length'] = self.predictions

        # 显示前5个预测结果
        formulas = self.data['formula'] if 'formula' in self.data.columns else [f"Sample {i + 1}" for i in range(len(self.predictions))]
        spacegroups = self.data['spacegroup.number'] if 'spacegroup.number' in self.data.columns else ["N/A"] * len(self.predictions)

        self.result_tree.delete(*self.result_tree.get_children())
        for formula, pred, spacegroup in zip(formulas, self.predictions, spacegroups):
            self.result_tree.insert("", "end", values=(formula, pred, spacegroup))

    def export_results(self):


        if self.data is None or self.predictions is None:
            messagebox.showwarning("Error", "Please run a prediction first")
            return

        # 检查 'Formula' 和 'Predicted H_learn' 是否在数据中
        if 'formula' not in self.data.columns:
            messagebox.showwarning("Error", "The 'formula' column is missing in the dataset")
            return
        if 'Predicted H_length' not in self.data.columns:
            messagebox.showwarning("Error", "The 'Predicted H_length' column is missing, please run predictions first")
            return

        # 弹出保存文件对话框
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            # 导出包含输入 formula 和预测结果的 CSV 文件
            export_columns = ['formula', 'Predicted H_length']
            if 'spacegroup.number' in self.data.columns:
                export_columns.append('spacegroup.number')
            self.data[export_columns].to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Results exported to {os.path.basename(file_path)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BorohydridePredictionTool(root)
    root.mainloop()
