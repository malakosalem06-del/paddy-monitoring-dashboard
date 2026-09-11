import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Paddy Field Monitoring & Yield Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background-color: #071923;
        color: #f4f7f8;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Remove Streamlit top spacing */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Main title */
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .main-subtitle {
        font-size: 16px;
        color: #a9c3ca;
        margin-bottom: 0px;
    }

    /* Header */
    .dashboard-header {
        background:
            linear-gradient(
                90deg,
                rgba(3, 27, 37, 0.98),
                rgba(3, 42, 43, 0.90),
                rgba(10, 65, 51, 0.72)
            );
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 12px;
        border: 1px solid #173b47;
        min-height: 115px;
    }

    .header-left {
        padding-top: 3px;
    }

    .leaf-symbol {
        font-size: 42px;
        color: #39c27a;
        font-weight: 700;
        float: left;
        margin-right: 18px;
        line-height: 1;
    }

    .header-tagline {
        float: right;
        text-align: right;
        color: #c9e8dc;
        font-size: 15px;
        line-height: 1.5;
        padding-top: 8px;
    }

    /* Navigation */
    .navigation {
        background-color: #0b2530;
        border-bottom: 2px solid #1fb77a;
        border-radius: 0 0 8px 8px;
        padding: 8px 12px;
        margin-bottom: 16px;
    }

    .nav-item {
        display: inline-block;
        padding: 11px 25px;
        margin-right: 5px;
        border-radius: 7px;
        color: #c3dbe0;
        font-size: 15px;
        font-weight: 500;
    }

    .nav-active {
        background-color: #17a878;
        color: white;
    }

    /* Section cards */
    .section-card {
        background: #0b202b;
        border: 1px solid #173d4b;
        border-radius: 15px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #f5f8fa;
        margin-bottom: 2px;
    }

    .section-subtitle {
        color: #9cb5bc;
        font-size: 14px;
        margin-bottom: 16px;
    }

    .section-number {
        display: inline-flex;
        width: 37px;
        height: 37px;
        border-radius: 50%;
        align-items: center;
        justify-content: center;
        background: #18a875;
        color: white;
        font-size: 19px;
        font-weight: 700;
        margin-right: 10px;
        vertical-align: middle;
    }

    /* Metric cards */
    .metric-card {
        border-radius: 9px;
        padding: 15px 17px;
        min-height: 100px;
        border: 1px solid #24576a;
        background: #102c39;
    }

    .metric-green {
        border-color: #168e69;
        background: #0c3733;
    }

    .metric-blue {
        border-color: #236a91;
        background: #102f42;
    }

    .metric-purple {
        border-color: #7047a6;
        background: #292044;
    }

    .metric-red {
        border-color: #a63d4b;
        background: #3a2029;
    }

    .metric-yellow {
        border-color: #9d8a34;
        background: #39351d;
    }

    .metric-label {
        color: #b7d3da;
        font-size: 14px;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 27px;
        font-weight: 700;
        line-height: 1.1;
    }

    .metric-unit {
        color: #a9c1c8;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Information box */
    .info-box {
        background: #12372f;
        border: 1px solid #237b62;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        color: #d8eee7;
        font-size: 13px;
    }

    /* Warning box */
    .warning-box {
        background: #3a2029;
        border: 1px solid #a83e4d;
        border-radius: 8px;
        padding: 15px;
        color: #f5d6da;
        min-height: 112px;
    }

    .warning-title {
        color: #ff7d88;
        font-weight: 700;
        margin-bottom: 6px;
        font-size: 15px;
    }

    /* Chart cards */
    .chart-card {
        background: #0e2935;
        border: 1px solid #1d4a59;
        border-radius: 10px;
        padding: 10px;
    }

    /* Image analysis */
    .image-card {
        background: #0d2834;
        border: 1px solid #1d4b5b;
        border-radius: 10px;
        padding: 14px;
    }

    .image-card-title {
        color: #61c7ed;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* Class legend */
    .legend-box {
        background: #102d39;
        border: 1px solid #285364;
        border-radius: 9px;
        padding: 16px;
        height: 100%;
    }

    .legend-title {
        color: #e5f0f3;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .legend-item {
        margin-bottom: 12px;
        color: #d3e0e3;
        font-size: 13px;
    }

    .legend-color {
        width: 17px;
        height: 17px;
        display: inline-block;
        border-radius: 4px;
        margin-right: 8px;
        vertical-align: middle;
    }

    .paddy-color {
        background: #ffff00;
    }

    .weed-color {
        background: #ff2020;
    }

    .background-color {
        background: #2e8b57;
    }

    /* Yield result */
    .yield-result {
        background: #0d3b2d;
        border: 1px solid #25a46e;
        border-radius: 10px;
        padding: 22px;
        text-align: center;
        min-height: 150px;
    }

    .yield-label {
        color: #c5e8d9;
        font-size: 15px;
        margin-bottom: 10px;
    }

    .yield-number {
        color: #ffffff;
        font-size: 31px;
        font-weight: 700;
    }

    /* Field cards */
    .field-card {
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        min-height: 92px;
        border: 1px solid #24596a;
        background: #102c38;
    }

    .field-label {
        font-size: 13px;
        color: #abc6cd;
        margin-bottom: 8px;
    }

    .field-value {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
    }

    /* Inputs */
    div[data-testid="stNumberInput"] label {
        color: #bcd3d9 !important;
        font-size: 13px !important;
    }

    div[data-baseweb="input"] {
        background-color: #102b37;
        border-radius: 6px;
    }

    div[data-baseweb="input"] input {
        color: white;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background: #102b37;
        border: 1px dashed #557581;
        border-radius: 8px;
    }

    /* Footer */
    .footer {
        border-top: 1px solid #244652;
        margin-top: 20px;
        padding-top: 18px;
        color: #91aab1;
        font-size: 13px;
        text-align: center;
    }

    /* Hide default menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="header-left">
            <div class="leaf-symbol">◆</div>
            <div class="main-title">
                Paddy Field Monitoring & Yield Prediction
            </div>
            <div class="main-subtitle">
                AI-powered segmentation and yield estimation for smarter rice farming
            </div>
        </div>

        <div class="header-tagline">
            Monitor<br>
            Analyse<br>
            Grow
        </div>
    </div>

    <div class="navigation">
        <span class="nav-item nav-active">Home</span>
        <span class="nav-item">Model Information</span>
        <span class="nav-item">About</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SEGMENTATION MODEL
# ============================================================

class MobileNetV2UNet(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = mobilenet_v2(weights=None)

        self.encoder = backbone.features

        self.up1 = nn.ConvTranspose2d(
            1280,
            320,
            2,
            stride=2
        )

        self.conv1 = nn.Conv2d(
            384,
            320,
            3,
            padding=1
        )

        self.up2 = nn.ConvTranspose2d(
            320,
            64,
            2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            96,
            64,
            3,
            padding=1
        )

        self.up3 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            56,
            32,
            3,
            padding=1
        )

        self.up4 = nn.ConvTranspose2d(
            32,
            16,
            2,
            stride=2
        )

        self.conv4 = nn.Conv2d(
            40,
            16,
            3,
            padding=1
        )

        self.up5 = nn.ConvTranspose2d(
            16,
            16,
            2,
            stride=2
        )

        self.final = nn.Conv2d(
            16,
            3,
            1
        )


    def forward(self, x):

        skip1 = None
        skip2 = None
        skip3 = None
        skip4 = None

        for i, layer in enumerate(self.encoder):

            x = layer(x)

            if i == 7:
                skip1 = x

            elif i == 6:
                skip2 = x

            elif i == 3:
                skip3 = x

            elif i == 2:
                skip4 = x


        x = self.up1(x)

        skip1 = F.interpolate(
            skip1,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip1],
            dim=1
        )

        x = self.conv1(x)


        x = self.up2(x)

        skip2 = F.interpolate(
            skip2,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip2],
            dim=1
        )

        x = self.conv2(x)


        x = self.up3(x)

        skip3 = F.interpolate(
            skip3,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip3],
            dim=1
        )

        x = self.conv3(x)


        x = self.up4(x)

        skip4 = F.interpolate(
            skip4,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip4],
            dim=1
        )

        x = self.conv4(x)


        x = self.up5(x)

        x = self.final(x)

        return x


# ============================================================
# LOAD SEGMENTATION MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MobileNetV2UNet().to(device)

model_path = "best_mobilenetv2_unet_improved_3class.pth"


try:

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model.eval()

except Exception as e:

    st.error(
        "The segmentation model could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# SECTION 1
# ============================================================

st.markdown(
    """
    <div class="section-card">

        <div class="section-title">
            <span class="section-number">1</span>
            Segmentation Model Performance
        </div>

        <div class="section-subtitle">
            Performance metrics on the unseen test dataset
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


metric_cols = st.columns(5)


metrics = [
    ("Paddy IoU", "53.96%", "metric-green"),
    ("Weed IoU", "58.69%", "metric-red"),
    ("Background IoU", "89.01%", "metric-blue"),
    ("Pixel Accuracy", "89.85%", "metric-purple"),
    ("Paddy Recall", "86.90%", "metric-yellow")
]


for col, metric in zip(metric_cols, metrics):

    with col:

        st.markdown(
            f"""
            <div class="metric-card {metric[2]}">
                <div class="metric-label">
                    {metric[0]}
                </div>

                <div class="metric-value">
                    {metric[1]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown(
    """
    <div class="info-box">
        <b>What this means:</b>
        The model achieves 89.85% overall pixel accuracy,
        with the strongest segmentation performance for background
        and moderate IoU for paddy and weeds.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD XGBOOST DATA
# ============================================================

data_path = "rice_xgboost_training_data_combined.csv"


try:

    yield_data = pd.read_csv(
        data_path
    )

except Exception as e:

    st.error(
        "The XGBoost dataset could not be loaded."
    )

    st.code(str(e))

    st.stop()


features = [
    "avg_height_cm",
    "rainfall_mm",
    "avg_temp_c"
]

target = "actual_yield_mt_ha"


required_columns = features + [
    target,
    "spatial_group"
]


missing_columns = [
    column
    for column in required_columns
    if column not in yield_data.columns
]


if missing_columns:

    st.error(
        "Required columns are missing from the XGBoost dataset."
    )

    st.write(
        missing_columns
    )

    st.stop()


xgboost_data = yield_data[
    required_columns
].copy()

xgboost_data = xgboost_data.dropna()


X = xgboost_data[
    features
]

y = xgboost_data[
    target
]

groups = xgboost_data[
    "spatial_group"
]


# ============================================================
# TRAIN XGBOOST
# ============================================================

yield_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)


yield_model.fit(
    X,
    y
)


# ============================================================
# SPATIAL VALIDATION
# ============================================================

cv_results = []

unique_groups = groups.nunique()

if unique_groups >= 2:

    group_kfold = GroupKFold(
        n_splits=unique_groups
    )

    for train_index, test_index in group_kfold.split(
        X,
        y,
        groups
    ):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]


        cv_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )


        cv_model.fit(
            X_train,
            y_train
        )


        predictions = cv_model.predict(
            X_test
        )


        fold_mae = mean_absolute_error(
            y_test,
            predictions
        )


        fold_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )


        if len(y_test) > 1:

            fold_r2 = r2_score(
                y_test,
                predictions
            )

        else:

            fold_r2 = np.nan


        cv_results.append(
            {
                "MAE": fold_mae,
                "RMSE": fold_rmse,
                "R2": fold_r2
            }
        )


cv_results_df = pd.DataFrame(
    cv_results
)


mean_mae = cv_results_df["MAE"].mean()

mean_rmse = cv_results_df["RMSE"].mean()

valid_r2 = cv_results_df["R2"].dropna()

if len(valid_r2) > 0:
    mean_r2 = valid_r2.mean()
else:
    mean_r2 = np.nan


# ============================================================
# SECTION 2
# ============================================================

st.markdown(
    """
    <div class="section-card">

        <div class="section-title">
            <span class="section-number">2</span>
            Yield Prediction Model
        </div>

        <div class="section-subtitle">
            XGBoost regression model trained using real field observations
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


yield_cols = st.columns(4)


with yield_cols[0]:

    st.markdown(
        f"""
        <div class="metric-card metric-blue">
            <div class="metric-label">
                Training Observations
            </div>

            <div class="metric-value">
                {len(xgboost_data)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with yield_cols[1]:

    st.markdown(
        f"""
        <div class="metric-card metric-green">
            <div class="metric-label">
                Spatial Groups
            </div>

            <div class="metric-value">
                {unique_groups}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with yield_cols[2]:

    st.markdown(
        f"""
        <div class="metric-card metric-yellow">
            <div class="metric-label">
                Observed Yield Range
            </div>

            <div class="metric-value">
                {y.min():.2f} - {y.max():.2f}
            </div>

            <div class="metric-unit">
                MT/ha
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


locations = sorted(
    xgboost_data[
        "spatial_group"
    ].astype(str).unique()
)


with yield_cols[3]:

    st.markdown(
        f"""
        <div class="metric-card metric-blue">
            <div class="metric-label">
                Locations
            </div>

            <div class="metric-value"
                 style="font-size:16px; margin-top:12px;">
                {" · ".join(locations)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SECTION 3
# ============================================================

st.markdown(
    """
    <div class="section-card">

        <div class="section-title">
            <span class="section-number">3</span>
            Spatial Validation Results
        </div>

        <div class="section-subtitle">
            Leave-one-location-out cross-validation
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


validation_cols = st.columns(
    [1, 1, 1, 1.8]
)


with validation_cols[0]:

    st.markdown(
        f"""
        <div class="metric-card metric-blue">
            <div class="metric-label">
                Mean MAE
            </div>

            <div class="metric-value">
                {mean_mae:.3f}
            </div>

            <div class="metric-unit">
                MT/ha
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with validation_cols[1]:

    st.markdown(
        f"""
        <div class="metric-card metric-purple">
            <div class="metric-label">
                Mean RMSE
            </div>

            <div class="metric-value">
                {mean_rmse:.3f}
            </div>

            <div class="metric-unit">
                MT/ha
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with validation_cols[2]:

    r2_text = (
        f"{mean_r2:.3f}"
        if not np.isnan(mean_r2)
        else "N/A"
    )

    st.markdown(
        f"""
        <div class="metric-card metric-red">
            <div class="metric-label">
                Mean R²
            </div>

            <div class="metric-value">
                {r2_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with validation_cols[3]:

    st.markdown(
        """
        <div class="warning-box">

            <div class="warning-title">
                Important Note
            </div>

            The negative R² indicates that the model does not
            generalize well to unseen locations in the current
            spatial validation. This result should be interpreted
            cautiously because the current dataset contains only
            three spatial groups.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHARTS
# ============================================================

chart_col1, chart_col2 = st.columns(2)


with chart_col1:

    st.markdown(
        """
        <div class="chart-card">
            <b style="font-size:17px;">Validation Metrics</b>
            <br>
            <span style="color:#9cb5bc;font-size:13px;">
                Comparison of MAE and RMSE
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


    fig1, ax1 = plt.subplots(
        figsize=(6, 3.5)
    )

    ax1.set_facecolor("#0e2935")
    fig1.patch.set_facecolor("#0e2935")

    bars = ax1.bar(
        ["MAE", "RMSE"],
        [mean_mae, mean_rmse],
        width=0.55
    )

    ax1.set_ylabel(
        "Error (MT/ha)",
        color="white"
    )

    ax1.tick_params(
        colors="white"
    )

    ax1.spines[
        "bottom"
    ].set_color(
        "#66808a"
    )

    ax1.spines[
        "left"
    ].set_color(
        "#66808a"
    )

    ax1.spines[
        "top"
    ].set_visible(False)

    ax1.spines[
        "right"
    ].set_visible(False)

    ax1.grid(
        axis="y",
        alpha=0.2
    )

    for bar, value in zip(
        bars,
        [mean_mae, mean_rmse]
    ):

        ax1.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
            color="white",
            fontweight="bold"
        )

    st.pyplot(
        fig1,
        use_container_width=True
    )

    plt.close(fig1)


with chart_col2:

    st.markdown(
        """
        <div class="chart-card">
            <b style="font-size:17px;">Observed Yield Distribution</b>
            <br>
            <span style="color:#9cb5bc;font-size:13px;">
                Distribution of grain yield in the training dataset
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


    fig2, ax2 = plt.subplots(
        figsize=(6, 3.5)
    )

    ax2.set_facecolor("#0e2935")
    fig2.patch.set_facecolor("#0e2935")

    ax2.hist(
        y,
        bins=14,
        edgecolor="#071923"
    )

    ax2.set_xlabel(
        "Grain Yield (MT/ha)",
        color="white"
    )

    ax2.set_ylabel(
        "Count",
        color="white"
    )

    ax2.tick_params(
        colors="white"
    )

    ax2.spines[
        "bottom"
    ].set_color(
        "#66808a"
    )

    ax2.spines[
        "left"
    ].set_color(
        "#66808a"
    )

    ax2.spines[
        "top"
    ].set_visible(False)

    ax2.spines[
        "right"
    ].set_visible(False)

    ax2.grid(
        axis="y",
        alpha=0.2
    )

    st.pyplot(
        fig2,
        use_container_width=True
    )

    plt.close(fig2)


# ============================================================
# SECTION 4
# ============================================================

st.markdown(
    """
    <div class="section-card">

        <div class="section-title">
            <span class="section-number">4</span>
            Image Analysis & Yield Prediction
        </div>

        <div class="section-subtitle">
            Upload a paddy field image to get segmentation results
            and estimated yield
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

upload_col, results_col = st.columns(
    [0.9, 2.5]
)


with upload_col:

    st.markdown(
        """
        <div class="image-card">

            <div class="image-card-title">
                Upload Image
            </div>

            <p style="color:#bdd1d6;font-size:13px;">
                Choose a paddy field image
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload a paddy field image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )


    st.caption(
        "Supported formats: JPG, JPEG, PNG"
    )


with results_col:

    st.markdown(
        """
        <div class="image-card-title">
            Segmentation Results
        </div>
        """,
        unsafe_allow_html=True
    )


    if uploaded_file is None:

        st.info(
            "Upload an image to display the AI segmentation results."
        )

    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        input_image = image.resize(
            (256, 256)
        )


        image_tensor = transforms.ToTensor()(
            input_image
        ).unsqueeze(0).to(device)


        with torch.no_grad():

            output = model(
                image_tensor
            )

            prediction = torch.argmax(
                output,
                dim=1
            )


        pred_mask = prediction.squeeze().cpu().numpy()


        mask_resized = Image.fromarray(
            pred_mask.astype(np.uint8)
        ).resize(
            image.size,
            resample=Image.Resampling.NEAREST
        )


        mask_array = np.array(
            mask_resized
        )


        # Class mapping
        paddy_mask = mask_array == 0
        background_mask = mask_array == 1
        weed_mask = mask_array == 2


        # Percentages
        paddy_percentage = (
            paddy_mask.mean() * 100
        )

        weed_percentage = (
            weed_mask.mean() * 100
        )

        background_percentage = (
            background_mask.mean() * 100
        )


        # Pixel counts
        paddy_pixels = int(
            paddy_mask.sum()
        )

        weed_pixels = int(
            weed_mask.sum()
        )

        background_pixels = int(
            background_mask.sum()
        )


        # ====================================================
        # AI PREDICTION IMAGE
        # ====================================================

        segmentation = np.zeros_like(
            np.array(image)
        )


        # Paddy = Yellow
        segmentation[
            paddy_mask
        ] = [
            255,
            255,
            0
        ]


        # Weed = Red
        segmentation[
            weed_mask
        ] = [
            255,
            30,
            30
        ]


        # Background = Green
        segmentation[
            background_mask
        ] = [
            46,
            139,
            87
        ]


        # ====================================================
        # OVERLAY
        # ====================================================

        original_array = np.array(
            image
        )


        overlay = original_array.copy()

        alpha = 0.45


        overlay[
            paddy_mask
        ] = (
            alpha
            * segmentation[paddy_mask]
            +
            (1 - alpha)
            * overlay[paddy_mask]
        ).astype(np.uint8)


        overlay[
            weed_mask
        ] = (
            alpha
            * segmentation[weed_mask]
            +
            (1 - alpha)
            * overlay[weed_mask]
        ).astype(np.uint8)


        # Background gets a lighter green overlay
        overlay[
            background_mask
        ] = (
            0.20
            * segmentation[background_mask]
            +
            0.80
            * overlay[background_mask]
        ).astype(np.uint8)


        # ====================================================
        # DISPLAY
        # ====================================================

        image_cols = st.columns(
            [1, 1, 1, 0.7]
        )


        with image_cols[0]:

            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )


        with image_cols[1]:

            st.image(
                segmentation,
                caption="AI Prediction",
                use_container_width=True
            )


        with image_cols[2]:

            st.image(
                overlay,
                caption="Overlay",
                use_container_width=True
            )


        with image_cols[3]:

            st.markdown(
                """
                <div class="legend-box">

                    <div class="legend-title">
                        Class Colors
                    </div>

                    <div class="legend-item">
                        <span class="legend-color paddy-color"></span>
                        Paddy
                    </div>

                    <div class="legend-item">
                        <span class="legend-color weed-color"></span>
                        Weed
                    </div>

                    <div class="legend-item">
                        <span class="legend-color background-color"></span>
                        Background
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# FIELD ANALYSIS AND YIELD
# ============================================================

if uploaded_file is not None:

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    field_col, input_col, result_col = st.columns(
        [1.2, 1.7, 0.7]
    )


    # ========================================================
    # FIELD ANALYSIS
    # ========================================================

    with field_col:

        st.markdown(
            """
            <div class="image-card-title">
                Field Analysis
            </div>
            """,
            unsafe_allow_html=True
        )


        f1, f2, f3 = st.columns(3)


        with f1:

            st.markdown(
                f"""
                <div class="field-card">
                    <div class="field-label">
                        Paddy Area
                    </div>

                    <div class="field-value">
                        {paddy_percentage:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with f2:

            st.markdown(
                f"""
                <div class="field-card">
                    <div class="field-label">
                        Weed Area
                    </div>

                    <div class="field-value">
                        {weed_percentage:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with f3:

            st.markdown(
                f"""
                <div class="field-card">
                    <div class="field-label">
                        Background
                    </div>

                    <div class="field-value">
                        {background_percentage:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # YIELD INPUTS
    # ========================================================

    with input_col:

        st.markdown(
            """
            <div class="image-card-title">
                Yield Prediction Inputs
            </div>
            """,
            unsafe_allow_html=True
        )


        default_height = float(
            xgboost_data[
                "avg_height_cm"
            ].mean()
        )


        default_rainfall = float(
            xgboost_data[
                "rainfall_mm"
            ].mean()
        )


        default_temperature = float(
            xgboost_data[
                "avg_temp_c"
            ].mean()
        )


        input1, input2, input3 = st.columns(3)


        with input1:

            height_input = st.number_input(
                "Average Crop Height (cm)",
                min_value=0.0,
                max_value=300.0,
                value=round(
                    default_height,
                    1
                ),
                step=0.1
            )


        with input2:

            rainfall_input = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                max_value=5000.0,
                value=round(
                    default_rainfall,
                    1
                ),
                step=1.0
            )


        with input3:

            temperature_input = st.number_input(
                "Average Temperature (°C)",
                min_value=0.0,
                max_value=50.0,
                value=round(
                    default_temperature,
                    1
                ),
                step=0.1
            )


    # ========================================================
    # YIELD PREDICTION
    # ========================================================

    with result_col:

        prediction_input = pd.DataFrame(
            {
                "avg_height_cm": [
                    height_input
                ],

                "rainfall_mm": [
                    rainfall_input
                ],

                "avg_temp_c": [
                    temperature_input
                ]
            }
        )


        predicted_yield = yield_model.predict(
            prediction_input
        )[0]


        predicted_yield = max(
            0,
            float(predicted_yield)
        )


        st.markdown(
            f"""
            <div class="yield-result">

                <div class="yield-label">
                    Estimated Yield
                </div>

                <div class="yield-number">
                    {predicted_yield:.2f}
                </div>

                <div style="
                    color:#bfe4d4;
                    font-size:14px;
                    margin-top:4px;
                ">
                    MT/ha
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-card">

        <div class="section-title">
            Model Information
        </div>

        <div class="section-subtitle">
            Current system configuration
        </div>

        <div style="
            color:#b7cbd0;
            font-size:14px;
            line-height:1.7;
        ">

        <b>Segmentation:</b>
        MobileNetV2-U-Net semantic segmentation model.

        <br>

        <b>Segmentation Classes:</b>
        Paddy, Weed, Background.

        <br>

        <b>Yield Model:</b>
        XGBoost regression.

        <br>

        <b>Yield Prediction Features:</b>
        Average crop height, rainfall, and average temperature.

        <br>

        <b>Training Dataset:</b>
        96 observations from three spatial groups:
        Akola, Jabalpur, and Faizabad.

        <br>

        <b>Spatial Validation:</b>
        Leave-one-location-out cross-validation.

        <br>

        <b>Important Limitation:</b>
        Paddy area and weed density are calculated from the
        segmentation model but are not currently used as
        XGBoost predictors because there are no paired yield
        observations containing these two features.

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Paddy Field Monitoring Dashboard
        &nbsp;&nbsp;|&nbsp;&nbsp;
        AI for Sustainable Rice Farming
    </div>
    """,
    unsafe_allow_html=True
)
