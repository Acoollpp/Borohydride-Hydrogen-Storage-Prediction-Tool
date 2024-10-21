import pandas as pd
import tkinter as tk
import os
from tkinter import filedialog, messagebox

from models import train, predictor
from features.feature_extraction import extract_features, extract_target

class BorohydridePredictionTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Borohydride Hydrogen Storage Property Prediction Tool")
        self.root.geometry("500x400")

        #设置导入数据按钮
        self.import_button = tk.Button(root, text="Import Data", command=self.import_data)
        self.import_button.pack(pady=10)

        # 模型选择下拉框
        self.model_var = tk.StringVar(root)
        self.model_var.set("Choose Model")  # 默认值
        self.model_dropdown = tk.OptionMenu(root, self.model_var, "Linear",
                                            "SVM",
                                            "Decision_tree",
                                            "Random_forest",
                                            "Lasso",
                                            'Ridge')
        self.model_dropdown.pack(pady=10)

        # 开始预测按钮
        self.predict_button = tk.Button(root, text="Start Prediction", command=self.run_prediction)
        self.predict_button.pack(pady=10)

        # 导出结果按钮
        self.export_button = tk.Button(root, text="Export Results", command=self.export_results)
        self.export_button.pack(pady=10)

        # 结果显示区域
        self.result_label = tk.Label(root, text="", wraplength=400)
        self.result_label.pack(pady=20)

        self.data = None
        self.model = None

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

        data_train = pd.read_csv('../data/raw/train_data.csv')
        features_train = extract_features(data_train)
        target = extract_target(data_train)

        # 检查是否选择了模型
        model_name = self.model_var.get()
        if model_name == "Choose Model":
            messagebox.showwarning("Error", "Please choose a model")
            return

        # 根据选择的模型进行预测
        if model_name == "Linear":
            model = train.linear(features_train, target)
        elif model_name == "SVM":
            # 假设有一个SVM模型训练函数
            model = train.svm(features_train, self.data['target'])
        elif model_name == "Decision_tree":
            # 决策树
            model = train.decision_tree(features_train, self.data['target'])
        elif model_name == "Random_forest":
            model = train.random_forest(features_train, self.data['target'])
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
        predictions_str = '\n'.join([f'Prediction {i + 1}: {pred}' for i, pred in enumerate(self.predictions)])
        self.result_label.config(text=f"Predictions:\n{predictions_str}")

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
            self.data[['formula', 'Predicted H_length']].to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Results exported to {os.path.basename(file_path)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BorohydridePredictionTool(root)
    root.mainloop()