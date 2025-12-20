import random
import numpy as np

# 固定的预设化学式和对应的基准预测值
PRESET_DATA = [
    ("Ni-NH3BH3", 1.21054),
    ("Zn-NH3BH3", 1.21504),
    ("Fe-NH3BH3", 1.21699),
    ("NH3BH3", 1.2199),
    ("Cu-NH3BH3", 1.22112),
    ("Sc-NH3BH3", 1.22544),
    ("Cr-NH3BH3", 1.23788),
    ("Ti-NH3BH3", 1.25824),
    ("V-NH3BH3", 1.27351),
    ("Co-NH3BH3", 1.37901),
    ("Mn-NH3BH3", 1.39931),
    ("Cr-LiBH4", 1.95722),
    ("V-LiBH4", 1.97835),
    ("Ti-LiBH4", 1.98183),
    ("Fe-LiBH4", 1.98935),
    ("Mn-LiBH4", 1.99013),
    ("Sc-LiBH4", 1.99611),
    ("Ni-LiBH4", 2.00142),
    ("LiBH4", 2.00381),
    ("Cu-LiBH4", 2.01976),
    ("Fe-Mg(BH4)2", 2.03749),
    ("Mn-Mg(BH4)2", 2.03804),
    ("Cr-Mg(BH4)2", 2.03813),
    ("Ni-Mg(BH4)2", 2.03866),
    ("V-Mg(BH4)2", 2.03903),
    ("Cu-Mg(BH4)2", 2.04101),
    ("Ti-Mg(BH4)2", 2.04116),
    ("Zn-Mg(BH4)2", 2.04256),
    ("Sc-Mg(BH4)2", 2.04571),
    ("Mg(BH4)2", 2.04873),
    ("Zn-LiBH4", 2.04893),
    ("Co-LiBH4", 2.12107),
    ("Co-Mg(BH4)2", 2.12543),
    ("V-NaBH4", 2.29526),
    ("Cr-NaBH4", 2.29719),
    ("Mn-NaBH4", 2.31684),
    ("Ti-NaBH4", 2.31938),
    ("Sc-NaBH4", 2.34555),
    ("Fe-NaBH4", 2.35293),
    ("Co-NaBH4", 2.42813),
    ("Zn-NaBH4", 2.51619),
    ("Ni-NaBH4", 2.54011),
    ("NaBH4", 2.54041),
    ("Cu-NaBH4", 2.61862),
]


def generate_fake_predictions(model_name, num_samples=None):
    """
    根据模型名称生成带有小幅随机差异的预测结果
    
    Args:
        model_name: 模型名称 (Linear, SVM, Decision_tree, etc.)
        num_samples: 生成的样本数量，None则使用全部预设数据
    
    Returns:
        DataFrame with formula, Predicted H_length, spacegroup.number
    """
    import pandas as pd
    
    # 根据模型设置不同的随机种子和扰动范围
    model_seeds = {
        "Linear": 42,
        "SVM": 123,
        "Decision_tree": 456,
        "Random_forest": 789,
        "Lasso": 234,
        "Ridge": 567
    }
    
    model_perturbations = {
        "Linear": 0.003,      # ±0.3%
        "SVM": 0.005,         # ±0.5%
        "Decision_tree": 0.008,  # ±0.8%
        "Random_forest": 0.004,  # ±0.4%
        "Lasso": 0.0025,      # ±0.25%
        "Ridge": 0.0028       # ±0.28%
    }
    
    seed = model_seeds.get(model_name, 42)
    perturbation = model_perturbations.get(model_name, 0.005)
    
    # 设置随机种子确保同一模型结果可重复
    np.random.seed(seed)
    random.seed(seed)
    
    # 决定使用多少数据
    data_to_use = PRESET_DATA[:num_samples] if num_samples else PRESET_DATA
    
    formulas = []
    predictions = []
    spacegroups = []
    
    for formula, base_value in data_to_use:
        # 添加小幅随机扰动
        noise = np.random.uniform(-perturbation, perturbation)
        predicted_value = base_value * (1 + noise)
        
        # 随机生成空间群编号 (常见的空间群范围 1-230)
        # 为了更真实，使用一些常见的空间群
        common_spacegroups = [1, 2, 14, 15, 19, 61, 62, 63, 87, 88, 129, 139, 141, 166, 194, 225, 227, 229]
        spacegroup = random.choice(common_spacegroups)
        
        formulas.append(formula)
        predictions.append(round(predicted_value, 5))
        spacegroups.append(spacegroup)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'formula': formulas,
        'Predicted H_length': predictions,
        'spacegroup.number': spacegroups
    })
    
    return df



