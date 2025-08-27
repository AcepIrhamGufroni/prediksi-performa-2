import math
import joblib as jb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier import ConfusionMatrix, ROCAUC

# ---------------------------
# Load Models with Error Handling
# ---------------------------
try:
    MODELS = {
        "Decision Tree": jb.load("model/DecisionTreeClassifier.pkl"),
        "Random Forest": jb.load("model/RandomForestClassifier.pkl"),
        "SVM": jb.load("model/SupportVectorClassifier.pkl"),
        "XGBoost": jb.load("model/XGBClassifier.pkl"),
    }
except Exception as e:
    st.error(f"Error loading models: {e}")
    MODELS = {}

# ---------------------------
# Utility Functions
# ---------------------------
def check_file_validity(file):
    """Validate uploaded Excel file against expected datatypes."""
    try:
        df = pd.read_excel(file)
    except Exception as e:
        return False, [f"Error reading the uploaded file: {e}"]
    try:
        expected_dtypes = pd.read_csv("data/datatypes.csv", header=None, names=["column", "dtype"])
    except Exception as e:
        return False, [f"Error reading expected datatypes file: {e}"]

    missing_columns, conversion_errors = [], []
    for _, row in expected_dtypes.iterrows():
        col, exp_dtype = row["column"], row["dtype"]
        if col not in df.columns:
            missing_columns.append(col)
        else:
            try:
                df[col] = df[col].astype(exp_dtype)
            except Exception as e:
                conversion_errors.append(f"{col} (expected {exp_dtype}, error: {e})")

    extra_columns = [c for c in df.columns if c not in expected_dtypes["column"].values]
    errors = []
    if missing_columns: errors.append("Missing columns: " + ", ".join(missing_columns))
    if extra_columns: errors.append("Unexpected extra columns: " + ", ".join(extra_columns))
    if conversion_errors: errors.append("Mismatched data types: " + ", ".join(conversion_errors))

    return (False, errors) if errors else (True, df)

def check_columns(file):
    """Check if file columns match expected."""
    try:
        df = pd.read_excel(file)
        file_columns = df.columns.tolist()
        target_columns = pd.read_csv("data/columns.csv", header=None)[0].tolist()
        return file_columns == target_columns
    except Exception as e:
        st.error(f"Error checking columns: {e}")
        return False

def prepare_viz(df):
    """Create boxplots for numeric features."""
    try:
        fill_values = {"bidikmisi": "Tidak Bidikmisi","Pilihan 2": "Belum Memilih",
                       "Lulus pada Prodi": "Tidak Lulus","Lulus Pilihan": "Tidak Lulus"}
        df.fillna(fill_values, inplace=True)
        df["JK"] = df["JK"].map({"P": 0, "L": 1})
        df["bidikmisi"] = df["bidikmisi"].map({"Bidik Misi": 1}).fillna(0).astype(int)
        df[["IP Sem 1","IP Sem 2"]] = df[["IP Sem 1","IP Sem 2"]].apply(pd.to_numeric, errors="coerce")
        df["Rata_IP"] = df[["IP Sem 1", "IP Sem 2"]].mean(axis=1)

        df_a = df[["Ranking Sekolah","Nilai Mapel UN","X1","X2","X3","X4","X5","X6","X7","X8","X9"]]
        num_columns, num_rows = 3, math.ceil(len(df_a.columns)/3)
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows*5))
        axs = axs.flatten()
        for i, col in enumerate(df_a.columns):
            sns.boxplot(y=df_a[col], ax=axs[i])
            axs[i].set_title(f"Distribusi {col}", fontsize=10)
        for j in range(i+1, len(axs)): fig.delaxes(axs[j])
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during visualization: {e}")

def preprocess(df):
    """Preprocess dataset: clean, encode, and separate features/target."""
    try:
        df = df.drop(["nomor_pendaftaran","nama_siswa","npsn_sekolah","tanggal_lahir"], axis=1)
        df["bidikmisi"] = (df["bidikmisi"] == "Bidik Misi").astype(int)
        df.fillna({"Lulus pada Prodi": "GAGAL","Pilihan 1": "TIDAK ADA","Pilihan 2": "TIDAK ADA",
                   "Lulus Pilihan": "GAGAL","Predikat Semester 1": "Tidak Ada",
                   "Predikat Semester 2": "Tidak Ada"}, inplace=True)
        df["IP Sem 1"].fillna(df["IP Sem 1"].median(), inplace=True)
        df["IP Sem 2"].fillna(df["IP Sem 2"].median(), inplace=True)
        cat_col = df.select_dtypes(exclude=np.number).columns
        for col in cat_col:
            if df[col].nunique() > 10:
                df[col] = df[col].map(df[col].value_counts())
            else:
                le = LabelEncoder(); df[col] = le.fit_transform(df[col])
        df["Rata_IP"] = df[["IP Sem 1","IP Sem 2"]].mean(axis=1)
        df["Predikat"] = pd.cut(df["Rata_IP"], bins=[-1,1,2,3,4], labels=[1,2,3,4]).astype(int)
        df = df[df["Predikat"] != 0].reset_index(drop=True)
        df["Predikat"] = df["Predikat"] - 1
        drop_cols = ["Sekolah","Pilihan 1","Pilihan 2","XT","IP Sem 1","IP Sem 2","Rata_IP",
                     "Predikat Semester 1","Predikat Semester 2"]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")
        return df.drop(columns=["Predikat"]), df["Predikat"].astype("category")
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None, None

# ---------------------------
# Load Custom CSS
# ---------------------------
try:
    with open("style.css", encoding="utf-8") as f:
        st.write("<style>" + f.read() + "</style>", unsafe_allow_html=True)
except Exception as e:
    st.error(f"Error loading style.css: {e}")

# ---------------------------
# Streamlit Session State Init
# ---------------------------
for key, default in [("uploaded_file", None), ("df", None), ("menu", "home")]:
    if key not in st.session_state:
        st.session_state[key] = default

logo = "img/Logo-Universitas-Diponegoro-UNDIP-Format-PNG-CDR-EPS-1536x1228.png"

# ---------------------------
# Result Menu Logic
# ---------------------------
if st.session_state.get("menu") == "result":
    try:
        df = st.session_state.get("df")
        x, y = preprocess(df)
        if x is None or y is None:
            st.error("Preprocessing failed.")
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=st.session_state.get("test_size"), random_state=777, stratify=y
            )
            comparison = []
            for m_name in st.session_state.get("selected_models", []):
                model = MODELS.get(m_name)
                if model is None: continue
                y_pred = model.predict(x_test)
                comparison.append({"Model": m_name,
                                   "Accuracy": accuracy_score(y_test, y_pred),
                                   "F1 Score": f1_score(y_test, y_pred, average="weighted"),
                                   "y_pred": y_pred})
            if not comparison:
                st.error("No valid models.")
            else:
                comp_df = pd.DataFrame(comparison)
                best_model = comp_df.sort_values(["Accuracy","F1 Score"], ascending=False).iloc[0]
                st.write(f"Best Model: {best_model['Model']} - Acc: {best_model['Accuracy']:.2f}, F1: {best_model['F1 Score']:.2f}")
                for m_name in st.session_state.get("selected_models"):
                    with st.container(border=True):
                        score_col, cm_col, res_col = st.columns([2,5,2], vertical_alignment="center", gap="small")
                        if st.session_state.get("show_score"):
                            with score_col:
                                st.write(f"{m_name}: Acc={comp_df.loc[comp_df['Model']==m_name]['Accuracy'].values[0]:.2f}, "
                                         f"F1={comp_df.loc[comp_df['Model']==m_name]['F1 Score'].values[0]:.2f}")
                        if st.session_state.get("show_cm"):
                            try:
                                with cm_col:
                                    model_clone = clone(MODELS.get(m_name))
                                    cm = ConfusionMatrix(model_clone, classes=["0","1","2","3"])
                                    cm.fit(x_train.values, y_train.values)
                                    cm.score(x_test.values, y_test.values)
                                    cm_path = f"img/{m_name}_confusion.png"
                                    cm.show(outpath=cm_path)
                                    st.image(cm_path)
                                    st.markdown(f"```\n{classification_report(y_test, MODELS.get(m_name).predict(x_test))}\n```")
                                    if st.session_state.get("show_rocauc"):
                                        roc_clone = clone(MODELS.get(m_name))
                                        roc = ROCAUC(roc_clone, classes=["0","1","2","3"])
                                        roc.fit(x_train.values, y_train.values)
                                        roc.score(x_test.values, y_test.values)
                                        roc_path = f"img/{m_name}_roc.png"
                                        roc.show(outpath=roc_path)
                                        st.image(roc_path)
                            except Exception as e:
                                st.error(f"Error displaying CM/ROC: {e}")
                        if st.session_state.get("results_file"):
                            with res_col:
                                preds = MODELS.get(m_name).predict(x_test)
                                file_path = f"data/x_predicted_{m_name}.csv"
                                pd.DataFrame({"Predictions": preds}).to_csv(file_path, index=False)
                                with open(file_path, "rb") as f:
                                    st.download_button(label="Download", data=f, file_name=f"x_predicted_{m_name}.csv")
    except Exception as e:
        st.error(f"Error in result menu: {e}")
