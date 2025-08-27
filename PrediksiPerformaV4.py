import math
import joblib as jb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier import ConfusionMatrix, ROCAUC
import seaborn as sns

# ---------------------------
# Load models with error handling
# ---------------------------
try:
    MODELS = {
        "Decision Tree": jb.load("model/DecisionTreeClassifier.pkl"),
        "Random Forest": jb.load("model/RandomForestClassifier.pkl"),
        "SVM": jb.load("model/SupportVectorClassifier.pkl"),
        "XGBoost": jb.load("model/XGBClassifier.pkl"),
    }
except Exception as e:
    st.error("Error loading models: " + str(e))
    MODELS = {}


# ---------------------------
# Function: check_file_validity
# Validates that the uploaded Excel file has the expected columns and datatypes as in datatypes.csv.
# If valid, automatically converts each column to the correct datatype.
# Returns a tuple: (True, converted_dataframe) if valid, or (False, list_of_error_messages)
# ---------------------------
def check_file_validity(file):
    try:
        df = pd.read_excel(file)
    except Exception as e:
        return False, [f"Error reading the uploaded file: {e}"]
    try:
        # Adjust the path if necessary
        expected_dtypes = pd.read_csv(
            "data/datatypes.csv", header=None, names=["column", "dtype"]
        )
    except Exception as e:
        return False, [f"Error reading expected datatypes file: {e}"]

    missing_columns = []
    conversion_errors = []

    # For each expected column, if present, attempt to convert to the target dtype.
    for _, row in expected_dtypes.iterrows():
        col = row["column"]
        exp_dtype = row["dtype"]
        if col not in df.columns:
            missing_columns.append(col)
        else:
            try:
                df[col] = df[col].astype(exp_dtype)
            except Exception as e:
                conversion_errors.append(
                    f"{col} (expected {exp_dtype}, conversion error: {e})"
                )
    # Check for extra columns not expected
    extra_columns = [
        col for col in df.columns if col not in expected_dtypes["column"].values
    ]

    errors = []
    if missing_columns:
        errors.append("Missing columns: " + ", ".join(missing_columns))
    if extra_columns:
        errors.append("Unexpected extra columns: " + ", ".join(extra_columns))
    if conversion_errors:
        errors.append("Mismatched data types: " + ", ".join(conversion_errors))

    if errors:
        return False, errors
    # df.to_csv("data/converted_data.csv", index=False) # Debugging
    return True, df


# ---------------------------
# Function: check_columns (legacy, if needed elsewhere)
# ---------------------------
def check_columns(file):
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error("Error reading uploaded file for column check: " + str(e))
        return False
    try:
        file_columns = df.columns.tolist()
    except Exception as e:
        st.error("Error retrieving columns from file: " + str(e))
        return False
    try:
        target_columns = pd.read_csv("data/columns.csv", header=None)[0].tolist()
    except Exception as e:
        st.error("Error reading expected columns from data/columns.csv: " + str(e))
        return False
    return file_columns == target_columns


# ---------------------------
# Function: prepare_viz
# ---------------------------
def prepare_viz(df):
    try:
        fill_values = {
            "bidikmisi": "Tidak Bidikmisi",
            "Pilihan 2": "Belum Memilih",
            "Lulus pada Prodi": "Tidak Lulus",
            "Lulus Pilihan": "Tidak Lulus",
        }
        df.fillna(fill_values, inplace=True)
        df["JK"] = df["JK"].map({"P": 0, "L": 1})
        df["bidikmisi"] = df["bidikmisi"].map({"Bidik Misi": 1}).fillna(0).astype(int)
        df[["IP Sem 1", "IP Sem 2"]] = df[["IP Sem 1", "IP Sem 2"]].apply(
            pd.to_numeric, errors="coerce"
        )
        df["Rata_IP"] = df[["IP Sem 1", "IP Sem 2"]].mean(axis=1)
        df["Predikat"] = df["Rata_IP"].apply(
            lambda x: (
                1
                if 0 < x <= 1
                else 2 if 1 < x <= 2 else 3 if 2 < x <= 3 else 4 if 3 < x <= 4 else 0
            )
        )
        df["Lulus Pilihan"] = (
            df["Lulus Pilihan"].map({"pil_1": 2, "pil_2": 1}).fillna(0).astype(int)
        )
        df_a = df[
            [
                "Ranking Sekolah",
                "Nilai Mapel UN",
                "X1",
                "X2",
                "X3",
                "X4",
                "X5",
                "X6",
                "X7",
                "X8",
                "X9",
            ]
        ]
        num_columns = 3
        num_rows = math.ceil(len(df_a.columns) / num_columns)

        fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
        axs = axs.flatten()

        for i, column in enumerate(df_a.columns):
            sns.boxplot(y=df_a[column], ax=axs[i])
            axs[i].set_title(f"Distribusi data {column}", fontsize=10)

        for j in range(i + 1, len(axs)):
            if j < len(axs):
                fig.delaxes(axs[j])

        st.pyplot(fig)
    except Exception as e:
        st.error("Error during visualization preparation: " + str(e))


# ---------------------------
# Function: preprocess
# ---------------------------
def preprocess(df):
    try:
        df = df.drop(
            ["nomor_pendaftaran", "nama_siswa", "npsn_sekolah", "tanggal_lahir"], axis=1
        )
        df["bidikmisi"] = (df["bidikmisi"] == "Bidik Misi").astype(int)

        df.fillna(
            {
                "Lulus pada Prodi": "GAGAL",
                "Pilihan 1": "TIDAK ADA",
                "Pilihan 2": "TIDAK ADA",
                "Lulus Pilihan": "GAGAL",
                "Predikat Semester 1": "Tidak Ada",
                "Predikat Semester 2": "Tidak Ada",
            },
            inplace=True,
        )
        df["IP Sem 1"] = df["IP Sem 1"].fillna(df["IP Sem 1"].median())
        df["IP Sem 2"] = df["IP Sem 2"].fillna(df["IP Sem 2"].median())
        df["JK"] = df["JK"].astype(str)
        cat_col = df.select_dtypes(exclude=np.number).columns
        print(cat_col)
        encoders = {}
        for col in cat_col:
            if (
                df[col].nunique() > 10
            ):  # Apply frequency encoding for high-cardinality columns
                df[col] = df[col].map(df[col].value_counts())
            else:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col])

        def grading(Rata_IP):
            if 0 < Rata_IP <= 1:
                return 1
            if 1 < Rata_IP <= 2:
                return 2
            if 2 < Rata_IP <= 3:
                return 3
            if 3 < Rata_IP <= 4:
                return 4
            return 0

        df["Rata_IP"] = df[["IP Sem 1", "IP Sem 2"]].mean(axis=1)
        df["Predikat"] = df["Rata_IP"].apply(grading)
        df = df[(df["Predikat"] != 0)]
        df = df.reset_index(drop=True)
        df["Predikat"] = df["Predikat"] - 1

        df = df.drop(
            [
                "Sekolah",
                "Pilihan 1",
                "Pilihan 2",
                "XT",
                "IP Sem 1",
                "IP Sem 2",
                "Rata_IP",
                "Predikat Semester 1",
                "Predikat Semester 2",
            ],
            axis=1,
        )

        return df.drop(columns=["Predikat"]), df["Predikat"].astype("category")
    except Exception as e:
        st.error("Error during preprocessing: " + str(e))
        return None, None


# ---------------------------
# Main app code
# ---------------------------
logo = "img/Logo-Universitas-Diponegoro-UNDIP-Format-PNG-CDR-EPS-1536x1228.png"

# Load custom CSS
try:
    with open("style.css", encoding="utf-8") as f:
        st.write("<style>" + f.read() + "</style>", unsafe_allow_html=True)
except Exception as e:
    st.error("Error loading style.css: " + str(e))

# Initialize session state variables
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "df" not in st.session_state:
    st.session_state["df"] = None

# In this version, we no longer re-read the file if a converted DataFrame is stored.
if (
    st.session_state.get("uploaded_file") is not None
    and st.session_state.get("df") is None
):
    try:
        # Read the file and store it (it may later be replaced with the converted DataFrame)
        st.session_state["df"] = pd.read_excel(st.session_state.get("uploaded_file"))
    except Exception as e:
        st.error("Error reading uploaded file: " + str(e))
        st.session_state["df"] = None

if "menu" not in st.session_state:
    st.session_state["menu"] = "home"

main_container = st.empty()

# ---------------------------
# Home Menu
# ---------------------------
if st.session_state["menu"] == "home":
    try:
        nav_left, nav_center, nav_right = st.columns(3)
        with nav_center:
            st.image(logo)
        st.write('<div class="title-container">', unsafe_allow_html=True)
        st.write("<h1>PrediksiPerforma</h1>", unsafe_allow_html=True)
        st.write(
            "<h5>Analisis Prediksi Performa Akademik Mahasiswa dengan Machine Learning</h5>",
            unsafe_allow_html=True,
        )
        st.write("</div>", unsafe_allow_html=True)

        if st.button("Mulai"):
            st.session_state["menu"] = "upload"
            st.rerun()
        if st.button("Tentang"):
            st.session_state["menu"] = "about"
            st.rerun()
    except Exception as e:
        st.error("Error in home menu: " + str(e))

# ---------------------------
# About Menu
# ---------------------------
if st.session_state.get("menu") == "about":
    try:
        nav_left, nav_center, nav_right = st.columns(3)
        with nav_center:
            st.image(logo)
        st.write(
            """
            <div class="about-container" style="text-align:justify;">
                <h3>About PrediksiPerforma</h3>
                <p>
                    PrediksiPerforma: Exploratory Data Analysis Prediksi Performa Akademik Mahasiswa Menggunakan Algoritma Supervised Learning merupakan aplikasi berbasis internet yang menggunakan exploratory data analysis dengan memodelkan algoritma supervised learning dalam memprediksi kemampuan akademik mahasiswa untuk mendapatkan algoritma dengan tingkat akurasi terbaik. Aplikasi ini dikembangkan sebagai salah satu tujuan dari penelitian Prediksi Performa Akademik Mahasiswa Menggunakan Algoritma Supervised Learning.
                </p>
                <p>
                    Tujuan dari penelitian ini adalah mengolah data admisi SNMPTN yang digabungkan dengan data nilai Indeks Prestasi Kumulatif (IPK), sehingga dapat diproses menggunakan model pembelajaran mesin. Data yang digunakan diproses menggunakan metodologi pembelajaran mesin untuk menghasilkan suatu model. Model tersebut dievaluasi dan dilakukan proses perbandingan dalam rangka mencari algoritma yang menghasilkan nilai evaluasi tertinggi. Model yang dibuat bertujuan untuk memprediksi performa akademik seseorang dengan mengklasifikasikannya ke dalam empat kelas label. Klasifikasi label ditentukan berdasarkan rentang nilai IPK.
                </p>
                <p>
                    Algoritma yang digunakan untuk membuat model adalah algoritma supervised learning klasifikasi, di antaranya Decision Tree (DT), Random Forest (RF), Support Vector Machine (SVM), dan Extreme Gradient Boosting (XGB). Penelitian dilakukan dalam tiga skema berdasarkan persentase antara data latih dengan data uji.
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )
        if st.button("Kembali ke Home"):
            st.session_state["menu"] = "home"
            st.rerun()
    except Exception as e:
        st.error("Error in about menu: " + str(e))

# ---------------------------
# Upload Menu
# ---------------------------
if st.session_state.get("menu") == "upload":
    try:
        nav_left, nav_center, nav_right = st.columns(
            [3, 10, 15], vertical_alignment="center", gap="small"
        )
        with nav_left:
            st.image(logo)
        with nav_center:
            st.write("<h3>PrediksiPerforma</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            for _ in range(3):
                st.columns(1)

            st.write(
                "<h5><center>Pilih Data atau Drop File Disini</center></h5>",
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(label="", type=["xlsx"])
            if uploaded_file is not None:
                st.session_state["uploaded_file"] = uploaded_file

        if st.button("Process"):
            if st.session_state.get("uploaded_file") is None:
                st.error("No file uploaded. Please upload an Excel file.")
            else:
                valid, result = check_file_validity(st.session_state["uploaded_file"])
                if valid:
                    # Store the automatically converted DataFrame for later use.
                    st.session_state["df"] = result
                    st.success("File processed successfully!")
                    st.session_state["menu"] = "config"
                    st.rerun()
                else:
                    for err in result:
                        st.error("File validation error: " + err)
        if st.button("Kembali ke Home"):
            st.session_state["menu"] = "home"
            st.rerun()
    except Exception as e:
        st.error("Error during file upload process: " + str(e))

# ---------------------------
# Config Menu
# ---------------------------
if st.session_state.get("menu") == "config":
    try:
        uploaded_file = st.session_state.get("uploaded_file", None)
        # Use the converted DataFrame from session state.
        df = st.session_state.get("df")
        nav_left, nav_center, nav_right = st.columns(
            [3, 10, 15], vertical_alignment="center", gap="small"
        )
        with nav_left:
            st.image(logo)
        with nav_center:
            st.write("<h3>PrediksiPerforma</h3>", unsafe_allow_html=True)
        column_left, column_right = st.columns(
            2, vertical_alignment="center", gap="small"
        )
        with column_left:
            with st.container(border=True):
                st.write("<h3>Hasil Analisa Dokumen</h3>", unsafe_allow_html=True)
                col_left_sub, col_right_sub = st.columns(
                    2, vertical_alignment="center", gap="large"
                )
                with col_right_sub:
                    if df is not None:
                        st.write(df.isnull().sum())
                        st.write(f"Data Incomplete: {df.isnull().sum().sum()}")
                    else:
                        st.error("Dataframe not loaded.")
                with col_left_sub:
                    with st.container(border=True):
                        st.image("logo/file-file-type-svgrepo-com.svg")
                        if uploaded_file is not None:
                            st.write(uploaded_file.name)
                    st.write("Format Validity :white_check_mark:")
                    try:
                        check_column = check_columns(uploaded_file)
                        if check_column:
                            st.write("Column Validity :white_check_mark:")
                            if df is not None:
                                prepare_viz(df)
                                st.write(
                                    f"Data distribution: {df.shape[1]} columns, {df.shape[0]} rows"
                                )
                        else:
                            st.write("Column Validity :x:")
                    except Exception as e:
                        st.error(
                            "Error during column check and visualization: " + str(e)
                        )
        with column_right:
            st.write("<h3>Model Configuraton</h3>", unsafe_allow_html=True)
            col_right1, col_right2 = st.columns(
                2, vertical_alignment="top", gap="small"
            )
            with col_right1:
                st.write("<h4>Algoritma</h4>", unsafe_allow_html=True)
                selected_models = st.multiselect("Pilih Algoritma", list(MODELS.keys()))
            with col_right2:
                st.write("<h4>Result Option</h4>", unsafe_allow_html=True)
                show_score = st.checkbox("Test Score", help="Presentase Skor Pengujian")
                show_cm = st.checkbox(
                    "Confusion Matrix", help="Visualisasi hasil prediksi"
                )
                show_rocauc = st.checkbox(
                    "ROC-AUC Curve", help="Visualisasi ROC-AUC Curve"
                )
                results_file = st.checkbox(
                    "Results File", help="File dengan tambahan hasil prediksi"
                )
            col_right3, col_right4 = st.columns(
                2, vertical_alignment="top", gap="large"
            )
            with col_right3:
                st.write("<h4>Train-test Ratio</h4>", unsafe_allow_html=True)
                ratio = st.radio("Select Train-Test Split", ("60-40", "70-30", "80-20"))
                test_size = {"60-40": 0.4, "70-30": 0.3, "80-20": 0.2}[ratio]
            with col_right4:
                for _ in range(6):
                    st.write("")
                if st.button("Generate"):
                    st.session_state["selected_models"] = selected_models
                    st.session_state["show_score"] = show_score
                    st.session_state["show_cm"] = show_cm
                    st.session_state["show_rocauc"] = show_rocauc
                    st.session_state["results_file"] = results_file
                    st.session_state["test_size"] = test_size
                    st.session_state["menu"] = "result"
                    st.rerun()
        if st.button("Kembali ke Home"):
            st.session_state["menu"] = "home"
            st.rerun()
    except Exception as e:
        st.error("Error in config menu: " + str(e))

# ---------------------------
# Result Menu
# ---------------------------
if st.session_state.get("menu") == "result":
    try:
        # Use the converted DataFrame stored in session state.
        df = st.session_state.get("df")
        x, y = preprocess(df)
        # print(x.isna().sum())
        # print(y)
        if x is None or y is None:
            st.error("Preprocessing failed. Cannot proceed.")
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=st.session_state.get("test_size"),
                random_state=777,
                stratify=y,
            )

            # sample_index = y_train[y_train == 0].index[0]

            # x_test = pd.concat([x_test, x_train.loc[[sample_index]]])
            # y_test = pd.concat([y_test, y_train.loc[[sample_index]]])

            # x_train = x_train.drop(sample_index)
            # y_train = y_train.drop(sample_index)

            # x_train = x_train.reset_index(drop=True)
            # y_train = y_train.reset_index(drop=True)
            # x_test = x_test.reset_index(drop=True)
            # y_test = y_test.reset_index(drop=True)

            comparison = []
            for model_name in st.session_state.get("selected_models"):
                model = MODELS.get(model_name)
                if model is None:
                    st.error(f"Model {model_name} not found.")
                    continue
                y_pred = model.predict(x_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                comparison.append(
                    {
                        "Model": model_name,
                        "Accuracy": acc,
                        "F1 Score": f1,
                        "y_pred": y_pred,
                    }
                )
            if not comparison:
                st.error("No valid models to evaluate.")
            else:
                comparison_df = pd.DataFrame(comparison)
                best_model = comparison_df.loc[
                    comparison_df.sort_values(
                        ["Accuracy", "F1 Score"], ascending=False
                    ).index[0]
                ]

                nav_left, nav_center, nav_right = st.columns(
                    [3, 10, 15], vertical_alignment="center", gap="small"
                )
                with nav_left:
                    st.image(logo)
                with nav_center:
                    st.write("<h3>PrediksiPerforma</h3>", unsafe_allow_html=True)

                st.write(
                    "<div style='text-align: justify;'> <h3>Hasil Prediksi</h3> </div>",
                    unsafe_allow_html=True,
                )
                st.write(
                    f"""
                    <div style='text-align: justify;'>
                        Setelah melalui proses pre-processing, model berhasil memproses {df.shape[0]} data 
                        sehingga didapat hasil prediksi berdasarkan klasifikasi yang telah ditentukan. 
                        Algoritma dengan presentase skor uji tertinggi diraih oleh:
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                col_algo, col_accuracy, col_f1 = st.columns([5, 3, 5])
                with col_algo:
                    st.write(
                        "<div style='text-align: justify;'><h3>Algoritma</h3></div>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"<div style='text-align: justify;'><h1><span style='color:lime'>{best_model['Model']}</span></h1></div>",
                        unsafe_allow_html=True,
                    )
                with col_accuracy:
                    st.write(
                        "<div style='text-align: justify;'><h3>Akurasi</h3></div>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"<div style='text-align: justify;'><h4><span style='color:lime'>{best_model['Accuracy']:.2f}</span></h4></div>",
                        unsafe_allow_html=True,
                    )
                with col_f1:
                    st.write(
                        "<div style='text-align: justify;'><h3>F1-Score</h3></div>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"<div style='text-align: justify;'><h4><span style='color:lime'>{best_model['F1 Score']:.2f}</span></h4></div>",
                        unsafe_allow_html=True,
                    )

                for model_name in st.session_state.get("selected_models"):
                    with st.container(border=True):
                        score_column, cm_column, results_column = st.columns(
                            [2, 5, 2], vertical_alignment="center", gap="small"
                        )
                        if st.session_state.get("show_score"):
                            try:
                                with score_column:
                                    st.write(
                                        f"<div style='text-align: justify;'>{model_name}</div>",
                                        unsafe_allow_html=True,
                                    )
                                    st.write(
                                        f"<div style='text-align: justify;'>Accuracy: {comparison_df.loc[comparison_df['Model'] == model_name]['Accuracy'].values[0]:.2f}</div>",
                                        unsafe_allow_html=True,
                                    )
                                    st.write(
                                        f"<div style='text-align: justify;'>F1 Score: {comparison_df.loc[comparison_df['Model'] == model_name]['F1 Score'].values[0]:.2f}</div>",
                                        unsafe_allow_html=True,
                                    )
                            except Exception as e:
                                st.error("Error displaying model score: " + str(e))
                        if st.session_state.get("show_cm"):
                            try:
                                with cm_column:
                                    plt.clf()
                                    fig, ax = plt.subplots()
                                    cm = ConfusionMatrix(
                                        MODELS.get(model_name),
                                        classes=["0", "1", "2", "3"],
                                    )
                                    print(len(x_train))
                                    print(len(y_train))
                                    print(len(x_test))
                                    print(len(y_test))
                                    cm.fit(x_train, y_train)
                                    cm.score(x_test, y_test)
                                    outpath = f"img/{model_name}_confusion_matrix.png"
                                    cm.show(outpath=outpath)
                                    st.image(outpath)
                                    st.markdown(
                                        f"```\n{classification_report(y_test, MODELS.get(model_name).predict(x_test))}\n```"
                                    )
                            except Exception as e:
                                st.error("Error displaying confusion matrix: " + str(e))
                        if st.session_state.get("show_rocauc"):
                            try:
                                with cm_column:  # or correct container
                                    plt.clf()
                                    model_clone = clone(MODELS.get(model_name))
                                    roc = ROCAUC(model_clone, classes=["0", "1", "2", "3"])
                                    le = LabelEncoder()
                                    y_train = le.fit_transform(y_train)
                                    roc.fit(x_train, y_train)
                                    roc.score(x_test, y_test)
                                    roc_outpath = f"img/{model_name}_roc_auc.png"
                                    roc.show(outpath=roc_outpath)
                                    st.image(roc_outpath)
                            except Exception as e:
                                st.error("Error displaying ROC AUC: " + str(e))
                        if st.session_state.get("results_file"):
                            try:
                                with results_column:
                                    with st.container(border=True):
                                        st.image("logo/file-file-type-svgrepo-com.svg")
                                        st.write(f"x_predicted_{model_name}.csv")
                                        data = pd.DataFrame(
                                            {
                                                "Predictions": pd.Series(
                                                    MODELS.get(model_name).predict(
                                                        x_test
                                                    )
                                                )
                                            }
                                        )
                                        file_path = f"data/x_predicted_{model_name}.csv"
                                        data.to_csv(file_path, index=False)
                                    with open(file_path, "rb") as file:
                                        st.download_button(
                                            label="Download",
                                            data=file,
                                            file_name=f"x_predicted_{model_name}.csv",
                                        )
                            except Exception as e:
                                st.error(
                                    "Error generating or downloading results file: "
                                    + str(e)
                                )
        if st.button("Kembali ke Home"):
            st.session_state["menu"] = "home"
            st.rerun()
    except Exception as e:
        st.error("Error during model evaluation and result generation: " + str(e))
