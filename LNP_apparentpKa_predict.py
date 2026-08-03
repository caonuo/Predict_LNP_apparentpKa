# pKa脂质预测+SHAP解释 Streamlit程序
'''pKa三分类预测 + SHAP模型解释 完整修复版'''
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import json
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
import time

# ====================== 全局页面配置 ======================
st.set_page_config(page_title="pKa Lipid Prediction", layout="wide")
plt.rcParams["axes.unicode_minus"] = False

# ====================== 缓存加载模型（只加载一次） ======================
model_path = 'pKa_Best_RF_Model.pkl'
@st.cache_resource
def load_model():
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_model()

# ====================== 缓存加载特征配置 ======================
@st.cache_data
def load_config():
    config_path = 'pKa_feature.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config()
rdkit_features = config['rdkit_descriptors']
formulation_features = config['formulation_features']
categorical_features = config['categorical_features']

# 拼接得到完整特征序列，替代不存在的 selected\_features
feature_order = list(model.feature_names_in_)
print(model.feature_names_in_[:20])
print(len(model.feature_names_in_))
# 初始化RDKit计算器（全局只初始化一次）
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(rdkit_features)

# ====================== 页面标题 & 输入面板 ======================
st.title("🔬 Ionizable Lipid pKa Classification Prediction")
st.divider()

smiles = st.text_input('Ionizable lipid SMILES', placeholder="请输入可电离脂质SMILES结构式")

# 四组分占比分栏
col1, col2, col3, col4 = st.columns(4)
with col1:
    ion_content = st.number_input(
        'Ionizable lipid (%)',
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        help="可电离脂质摩尔占比"
    )
with col2:
    helper_content = st.number_input("Helper lipid (%)", value=10.0)
with col3:
    peg_content = st.number_input("PEG lipid (%)", value=1.5)
with col4:
    chol_content = st.number_input("Cholesterol (%)", value=38.5)

col5, col6 = st.columns(2)
with col5:
    buffer_ph = st.number_input("Buffer pH", value=4.0)
with col6:
    helper = st.selectbox("Helper lipid", ["DOPE", "DSPC"])

st.divider()

# ====================== 预测按钮逻辑（全部逻辑收在内部） ======================
if st.button("🚀 Start Prediction", type="primary"):
    start_time = time.time()
    with st.spinner("正在计算分子描述符、执行模型预测..."):
        try:
            # 1. SMILES空值校验
            if not smiles.strip():
                st.error("❌ 请输入可电离脂质的SMILES结构式！")
                st.stop()

            # 2. SMILES合法性校验
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.error("❌ 输入的SMILES无效，请检查结构式！")
                st.stop()

            # 3. RDKit分子特征计算
            values = calculator.CalcDescriptors(mol)
            rdkit_dict = {f'Ionizable_lipid_SMILES_{name}':value
                        for name,value in zip(rdkit_features, values)}

            # 4. 写入配方连续特征
            rdkit_dict["Ionizable_lipid_content"] = ion_content
            rdkit_dict["Helper_lipid_content"] = helper_content
            rdkit_dict["PEG_lipid_content"] = peg_content
            rdkit_dict["Cholesterol_content"] = chol_content
            rdkit_dict["Aqueous_Buffer_PH"] = buffer_ph

            # 5. 校验四组分总和是否100%
            total_ratio = ion_content + helper_content + peg_content + chol_content
            if abs(total_ratio - 100) > 0.01:
                st.warning(f"⚠️ 当前四种脂质配比总和 = {total_ratio:.2f} %，推荐总和为100%")

            # 6. 辅助脂质独热编码
            rdkit_dict["Helper_lipid_DOPE"] = 0
            rdkit_dict["Helper_lipid_DSPC"] = 0
            if helper == "DOPE":
                rdkit_dict["Helper_lipid_DOPE"] = 1
            elif helper == "DSPC":
                rdkit_dict["Helper_lipid_DSPC"] = 1

            # 7. 构造输入特征矩阵，对齐训练特征顺序
            X = pd.DataFrame([rdkit_dict])
            X = X.reindex(columns=feature_order, fill_value=0)

            # 8. 模型预测（三分类 0/1/2）
            predicted_class = model.predict(X)[0]
            predicted_proba = model.predict_proba(X)[0]

            # ====================== 预测结果展示 ======================
            st.success("✅ 预测完成！")
            st.subheader("📊 预测结果")
            class_map = {0: "pKa<6", 1: "6<pKa<7", 2: "pKa>7"}
            pred_label = class_map[int(predicted_class)]
            pred_prob = round(predicted_proba[int(predicted_class)] * 100, 2)

            st.write(f"**预测pKa类别：{pred_label}（类别 {int(predicted_class)}）**")
            st.write(f"**预测置信度：{pred_prob}%**")

            # 概率表格
            prob_df = pd.DataFrame({
                "pKa等级": ["pKa<6", "6<pKa<7", "pKa>7"],
                "类别编号": [0, 1, 2],
                "预测概率(%)": np.round(predicted_proba * 100, 2)
            })
            st.dataframe(prob_df, use_container_width=True)

            # 概率柱状图
            st.subheader("📈 各类别预测概率分布")
            st.bar_chart(prob_df.set_index("pKa等级")["预测概率(%)"], use_container_width=True)

            st.info(f"💡 综合判定：该脂质配方的pKa水平为【{pred_label}】，模型预测置信度为 {pred_prob}%")

            # ====================== SHAP特征解释 ======================
            st.divider()
            with st.expander("🔍 SHAP 模型特征解释（可展开）", expanded=True):
                shap_values = explainer(X)
                target_cls = int(predicted_class)
                exp = shap.Explanation(
                    values=shap_values.values[0, :, target_cls],
                    base_values=shap_values.base_values[0, target_cls],
                    data=X.iloc[0],
                    feature_names=X.columns
                )
                # 绘图规范写法，避免画布堆积
                fig, ax = plt.subplots(figsize=(12, 6))
                shap.plots.waterfall(exp, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("上图展示各特征对本次pKa预测结果的正向/负向贡献大小")

            # 统计耗时
            end_time = time.time()
            st.success(f"⏱️ 本次预测总耗时：{end_time - start_time:.2f} s")

        except Exception as e:
            st.error(f"程序运行出错：{str(e)}")
