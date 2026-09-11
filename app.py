
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

st.set_page_config(
    page_title="Paddy Field Monitoring & Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PAGE STYLE
# ============================================================

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

.stApp {
    background: #071923;
    color: #f4f7f8;
}

.block-container {
    max-width: 1500px;
    padding: 1.2rem 2rem 3rem 2rem;
}

.hero {
    background: linear-gradient(120deg, #08232c, #063c35);
    border: 1px solid #174b54;
    border-radius: 20px;
    padding: 28px 34px;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    margin: 0;
    color: #f6fafb;
}

.hero-subtitle {
    font-size: 19px;
    color: #a9c1c7;
    margin-top: 8px;
}

.hero-tag {
    float: right;
    text-align: center;
    font-size: 22px;
    line-height: 1.15;
    font-style: italic;
    color: #e8f2ed;
    margin-top: -55px;
}

.navbar {
    background: #0a2530;
    border-bottom: 2px solid #18b986;
    border-radius: 0 0 12px 12px;
    padding: 12px 20px;
    margin-bottom: 18px;
}

.section {
    background: linear-gradient(135deg, #0a202c, #0b1d28);
    border: 1px solid #193b49;
    border-radius: 18px;
    padding: 20px;
    margin: 14px 0;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 3px;
}

.section-subtitle {
    color: #91aeb7;
    font-size: 14px;
    margin-bottom: 16px;
}

.section-number {
    display: inline-flex;
    width: 38px;
    height: 38px;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #1dbb91;
    margin-right: 10px;
    font-size: 19px;
}

.card {
    background: #102c39;
    border: 1px solid #1d5667;
    border-radius: 12px;
    padding: 16px;
    min-height: 112px;
}

.card-green {
    border-color: #1ba77f;
    background: linear-gradient(135deg, #103a35, #102c39);
}

.card-blue {
    border-color: #2371ad;
}

.card-purple {
    border-color: #7640b5;
}

.card-red {
    border-color: #a93c4b;
    background: #351d27;
}

.card-label {
    color: #b9d3db;
    font-size: 14px;
    margin-bottom: 8px;
}

.card-value {
    color: #f7fbfc;
    font-size: 25px;
    font-weight: 800;
}

.card-small {
    color: #a7c0c7;
    font-size: 12px;
    margin-top: 5px;
}

.info-box {
    background: #10352f;
    border: 1px solid #1e9673;
    border-radius: 10px;
    padding: 13px 16px;
    margin-top: 10px;
    color: #d8ece7;
}

.warning-box {
    background: #38222a;
    border: 1px solid #a94352;
    border-radius: 10px;
    padding: 14px 16px;
    color: #f2d9de;
}

.image-panel {
    background: #0e2935;
    border: 1px solid #1b4d5e;
    border-radius: 12px;
    padding: 12px;
}

.legend {
    background: #0e2935;
    border: 1px solid #1b4d5e;
    border-radius: 12px;
    padding: 15px;
    height: 100%;
}

.legend-row {
    display: flex;
    align-items: center;
    margin: 10px 0;
}

.legend-color {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    margin-right: 10px;
}

.big-result {
    background: linear-gradient(135deg, #0e563e, #0d3e32);
    border: 1px solid #1cad7e;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    min-height: 150px;
}

.big-result-label {
    font-size: 15px;
    color: #c7e4db;
}

.big-result-value {
    font-size: 32px;
    font-weight: 800;
    margin-top: 12px;
}

div[data-testid="stFileUploader"] {
    background: #102c39;
    border: 1px solid #2a5b6b;
    border-radius: 12px;
    padding: 10px;
}

.stButton > button {
    border-radius: 10px;
}

div[data-testid="stMetric"] {
    background: #102c39;
    border: 1px solid #1d5667;
    padding: 12px;
    border-radius: 10px;
}

[data-testid="stTabs"] button {
    color: #b8cfd5;
    font-size: 16px;
    font-weight: 700;
}

hr {
    border-color: #21414c;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">◆ &nbsp; Paddy Field Monitoring &amp; Yield Prediction</div>
    <div class="hero-subtitle">AI-powered segmentation and yield estimation for smarter rice farming</div>
    <div class="hero-tag">Monitor<br>Analyse<br>Grow</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Home", "📊 Model Information", "ℹ️ About"])

# ============================================================
# SEGMENTATION MODEL
# ============================================================

class MobileNetV2UNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v2(weights=None)
        self.encoder = backbone.features

        self.up1 = nn.ConvTranspose2d(1280, 320, 2, stride=2)
        self.conv1 = nn.Conv2d(384, 320, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(320, 64, 2, stride=2)
        self.conv2 = nn.Conv2d(96, 64, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv3 = nn.Conv2d(56, 32, 3, padding=1)

        self.up4 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv4 = nn.Conv2d(40, 16, 3, padding=1)

        self.up5 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.final = nn.Conv2d(16, 3, 1)

    def forward(self, x):
        skip1 = skip2 = skip3 = skip4 = None

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
        skip1 = F.interpolate(skip1, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        skip2 = F.interpolate(skip2, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        skip3 = F.interpolate(skip3, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        skip4 = F.interpolate(skip4, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip4], dim=1)
        x = self.conv4(x)

        x = self.up5(x)
        x = self.final(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MobileNetV2UNet().to(device)
model_path = "best_mobilenetv2_unet_improved_3class.pth"

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    model_loaded = True

except Exception as e:
    model_loaded = False
    model_error = str(e)

# ============================================================
# LOAD XGBOOST DATASET
# ============================================================

data_path = "rice_xgboost_training_data_combined.csv"

try:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    yield_data = pd.read_csv(data_path)

    features = [
        "avg_height_cm",
        "rainfall_mm",
        "avg_temp_c"
    ]

    target = "actual_yield_mt_ha"

    required_columns = features + [target, "spatial_group"]

    missing_columns = [
        c for c in required_columns
        if c not in yield_data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    xgboost_data = yield_data[required_columns].copy().dropna()

    X = xgboost_data[features]
    y = xgboost_data[target]
    groups = xgboost_data["spatial_group"]

    yield_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    yield_model.fit(X, y)

    unique_groups = groups.nunique()
    cv_results = []

    if unique_groups >= 2:
        group_kfold = GroupKFold(n_splits=unique_groups)

        for train_index, test_index in group_kfold.split(X, y, groups):
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

            cv_model.fit(X_train, y_train)
            predictions = cv_model.predict(X_test)

            fold_mae = mean_absolute_error(y_test, predictions)
            fold_rmse = np.sqrt(mean_squared_error(y_test, predictions))

            if len(y_test) > 1:
                fold_r2 = r2_score(y_test, predictions)
            else:
                fold_r2 = np.nan

            cv_results.append({
                "MAE": fold_mae,
                "RMSE": fold_rmse,
                "R2": fold_r2
            })

    cv_results_df = pd.DataFrame(cv_results)

    if len(cv_results_df) > 0:
        mean_mae = cv_results_df["MAE"].mean()
        mean_rmse = cv_results_df["RMSE"].mean()
        valid_r2 = cv_results_df["R2"].dropna()
        mean_r2 = valid_r2.mean() if len(valid_r2) > 0 else np.nan
    else:
        mean_mae = mean_rmse = mean_r2 = np.nan

    locations = sorted(
        xgboost_data["spatial_group"].astype(str).unique()
    )

    dataset_loaded = True

except Exception as e:
    dataset_loaded = False
    dataset_error = str(e)

# ============================================================
# HOME TAB
# ============================================================

with tabs[0]:

    if not model_loaded:
        st.error("The AI segmentation model could not be loaded.")
        st.code(model_error)

    if not dataset_loaded:
        st.error("The XGBoost training dataset could not be loaded.")
        st.code(dataset_error)

    # --------------------------------------------------------
    # SECTION 1
    # --------------------------------------------------------

    st.markdown("""
    <div class="section">
        <div class="section-title">
            <span class="section-number">1</span>
            Segmentation Model Performance
        </div>
        <div class="section-subtitle">Performance metrics on the test dataset</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown('<div class="card card-green"><div class="card-label">🌾 Paddy IoU</div><div class="card-value">53.96%</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card card-blue"><div class="card-label">🌱 Weed IoU</div><div class="card-value">58.69%</div></div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card card-blue"><div class="card-label">▰ Background IoU</div><div class="card-value">89.01%</div></div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card card-blue"><div class="card-label">🎯 Pixel Accuracy</div><div class="card-value">89.85%</div></div>', unsafe_allow_html=True)

    with c5:
        st.markdown('<div class="card card-green"><div class="card-label">▥ Paddy Recall</div><div class="card-value">86.90%</div></div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
        <b>What this means:</b> The model achieves 89.85% overall pixel accuracy,
        with the strongest segmentation performance for background and moderate
        IoU for paddy and weeds.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # SECTION 2
    # --------------------------------------------------------

    st.markdown("""
    <div class="section">
        <div class="section-title">
            <span class="section-number">2</span>
            Yield Prediction Model
        </div>
        <div class="section-subtitle">
            XGBoost regression model trained on the available field yield dataset
        </div>
    """, unsafe_allow_html=True)

    if dataset_loaded:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="card card-blue">
                <div class="card-label">🗄 Training Observations</div>
                <div class="card-value">{len(xgboost_data)}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="card card-blue">
                <div class="card-label">📍 Spatial Groups</div>
                <div class="card-value">{unique_groups}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="card card-green">
                <div class="card-label">📈 Observed Yield Range</div>
                <div class="card-value">{y.min():.2f} – {y.max():.2f}</div>
                <div class="card-small">MT/ha</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            location_text = " • ".join(locations)
            st.markdown(f"""
            <div class="card card-blue">
                <div class="card-label">📍 Locations</div>
                <div class="card-value" style="font-size:17px;">{location_text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SECTION 3
    # --------------------------------------------------------

    st.markdown("""
    <div class="section">
        <div class="section-title">
            <span class="section-number">3</span>
            Spatial Validation Results
        </div>
        <div class="section-subtitle">
            Leave-one-location-out cross-validation
        </div>
    """, unsafe_allow_html=True)

    if dataset_loaded and len(cv_results_df) > 0:

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="card card-blue">
                <div class="card-label">MAE</div>
                <div class="card-value">{mean_mae:.3f}</div>
                <div class="card-small">MT/ha</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="card card-purple">
                <div class="card-label">RMSE</div>
                <div class="card-value">{mean_rmse:.3f}</div>
                <div class="card-small">MT/ha</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            r2_text = "N/A" if np.isnan(mean_r2) else f"{mean_r2:.3f}"
            st.markdown(f"""
            <div class="card card-red">
                <div class="card-label">R²</div>
                <div class="card-value">{r2_text}</div>
            </div>
            """, unsafe_allow_html=True)

        if not np.isnan(mean_r2) and mean_r2 < 0:
            st.markdown(f"""
            <div class="warning-box">
                <b>Note:</b> The negative R² indicates that the current model
                does not generalize well to the unseen locations in this spatial
                validation. This reflects the limited number of spatial groups
                currently available in the dataset.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        chart1, chart2 = st.columns(2)

        with chart1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(["MAE", "RMSE"], [mean_mae, mean_rmse])
            ax.set_ylabel("Error (MT/ha)")
            ax.set_title("Validation Metrics")
            ax.grid(axis="y", alpha=0.25)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with chart2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(y, bins=10)
            ax.set_xlabel("Grain Yield (MT/ha)")
            ax.set_ylabel("Count")
            ax.set_title("Observed Yield Distribution")
            ax.grid(axis="y", alpha=0.25)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown(
            "Spatial validation uses the available locations as groups so that "
            "observations from the same location are not mixed between training "
            "and validation.",
        )

    else:
        st.warning("Spatial validation could not be performed.")

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SECTION 4
    # --------------------------------------------------------

    st.markdown("""
    <div class="section">
        <div class="section-title">
            <span class="section-number">4</span>
            Image Analysis &amp; Yield Prediction
        </div>
        <div class="section-subtitle">
            Upload a paddy field image to get segmentation results and estimated yield
        </div>
    """, unsafe_allow_html=True)

    upload_col, image_col = st.columns([1, 3])

    with upload_col:
        st.markdown('<div class="image-panel">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Image")
        st.write("Choose a paddy field image.")

        uploaded_file = st.file_uploader(
            "Drag and drop here or click to browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible"
        )

        st.caption("Supports: JPG, JPEG, PNG")
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None and model_loaded:

        image = Image.open(uploaded_file).convert("RGB")

        input_image = image.resize((256, 256))
        image_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1)

        pred_mask = prediction.squeeze().cpu().numpy()

        mask_resized = Image.fromarray(
            pred_mask.astype(np.uint8)
        ).resize(
            image.size,
            resample=Image.Resampling.NEAREST
        )

        mask_array = np.array(mask_resized)

        paddy_mask = mask_array == 0
        background_mask = mask_array == 1
        weed_mask = mask_array == 2

        paddy_percentage = paddy_mask.mean() * 100
        background_percentage = background_mask.mean() * 100
        weed_percentage = weed_mask.mean() * 100

        total_pixels = mask_array.size
        paddy_pixels = int(paddy_mask.sum())
        weed_pixels = int(weed_mask.sum())
        background_pixels = int(background_mask.sum())

        original_array = np.array(image)

        # Paddy = yellow
        # Weed = red
        # Background = green
        segmentation = np.zeros_like(original_array)
        segmentation[paddy_mask] = [255, 255, 0]
        segmentation[background_mask] = [46, 160, 100]
        segmentation[weed_mask] = [255, 0, 0]

        overlay = original_array.copy()
        alpha = 0.45

        overlay[paddy_mask] = (
            alpha * segmentation[paddy_mask]
            + (1 - alpha) * overlay[paddy_mask]
        ).astype(np.uint8)

        overlay[background_mask] = (
            alpha * segmentation[background_mask]
            + (1 - alpha) * overlay[background_mask]
        ).astype(np.uint8)

        overlay[weed_mask] = (
            alpha * segmentation[weed_mask]
            + (1 - alpha) * overlay[weed_mask]
        ).astype(np.uint8)

        with image_col:
            st.markdown('<div class="image-panel">', unsafe_allow_html=True)
            st.markdown("### ⚙️ Segmentation Results")

            image1, image2, image3, legend_col = st.columns([1, 1, 1, 0.55])

            with image1:
                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )

            with image2:
                st.image(
                    segmentation,
                    caption="AI Prediction",
                    use_container_width=True
                )

            with image3:
                st.image(
                    overlay,
                    caption="Segmentation Overlay",
                    use_container_width=True
                )

            with legend_col:
                st.markdown("""
                <div class="legend">
                    <b>Class Colors</b>
                    <div class="legend-row">
                        <div class="legend-color" style="background:#ffff00;"></div>
                        Paddy
                    </div>
                    <div class="legend-row">
                        <div class="legend-color" style="background:#ff0000;"></div>
                        Weed
                    </div>
                    <div class="legend-row">
                        <div class="legend-color" style="background:#2ea064;"></div>
                        Background
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # FIELD ANALYSIS
        # ----------------------------------------------------

        field_col, input_col, result_col = st.columns([1.25, 1.45, 0.75])

        with field_col:
            st.markdown("""
            <div class="card">
                <div class="card-label">🌿 Field Analysis (from segmentation)</div>
            </div>
            """, unsafe_allow_html=True)

            f1, f2, f3 = st.columns(3)

            with f1:
                st.metric("Paddy Area", f"{paddy_percentage:.2f}%")

            with f2:
                st.metric("Weed Area", f"{weed_percentage:.2f}%")

            with f3:
                st.metric("Background", f"{background_percentage:.2f}%")

        with input_col:
            st.markdown("""
            <div class="card">
                <div class="card-label">⚙️ Yield Prediction Inputs</div>
            </div>
            """, unsafe_allow_html=True)

            mean_height = float(xgboost_data["avg_height_cm"].mean())
            mean_rainfall = float(xgboost_data["rainfall_mm"].mean())
            mean_temperature = float(xgboost_data["avg_temp_c"].mean())

            i1, i2, i3 = st.columns(3)

            with i1:
                height_input = st.number_input(
                    "Average Crop Height (cm)",
                    min_value=0.0,
                    max_value=300.0,
                    value=round(mean_height, 2),
                    step=0.1
                )

            with i2:
                rainfall_input = st.number_input(
                    "Rainfall (mm)",
                    min_value=0.0,
                    max_value=5000.0,
                    value=round(mean_rainfall, 2),
                    step=1.0
                )

            with i3:
                temperature_input = st.number_input(
                    "Average Temperature (°C)",
                    min_value=0.0,
                    max_value=50.0,
                    value=round(mean_temperature, 2),
                    step=0.1
                )

        with result_col:
            prediction_input = pd.DataFrame({
                "avg_height_cm": [height_input],
                "rainfall_mm": [rainfall_input],
                "avg_temp_c": [temperature_input]
            })

            predicted_yield = yield_model.predict(prediction_input)[0]

            st.markdown(f"""
            <div class="big-result">
                <div class="big-result-label">📈 Estimated Yield</div>
                <div class="big-result-value">{predicted_yield:.2f} MT/ha</div>
            </div>
            """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # PIXEL + GSD INFORMATION
        # ----------------------------------------------------

        with st.expander("🔎 Segmentation Pixel Information & Physical Paddy Area"):
            p1, p2, p3 = st.columns(3)

            with p1:
                st.metric("Paddy Pixels", f"{paddy_pixels:,}")

            with p2:
                st.metric("Weed Pixels", f"{weed_pixels:,}")

            with p3:
                st.metric("Background Pixels", f"{background_pixels:,}")

            st.write(
                "Physical paddy area = paddy pixels × (GSD in metres/pixel)². "
                "Enter the actual image GSD to calculate a physically meaningful area."
            )

            gsd_cm = st.number_input(
                "GSD (cm per pixel)",
                min_value=0.01,
                max_value=100.0,
                value=None,
                step=0.01,
                format="%.2f"
            )

            if gsd_cm is not None:
                gsd_m = gsd_cm / 100.0
                pixel_area_m2 = gsd_m ** 2
                paddy_area_m2 = paddy_pixels * pixel_area_m2

                st.metric(
                    "Estimated Paddy Area",
                    f"{paddy_area_m2:.2f} m²"
                )
            else:
                paddy_area_m2 = None
                st.info("Enter the actual GSD to calculate physical paddy area.")

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.markdown("### 📋 Field Monitoring Summary")

        summary_lines = [
            f"**Paddy coverage:** {paddy_percentage:.2f}%",
            f"**Weed coverage:** {weed_percentage:.2f}%",
            f"**Background coverage:** {background_percentage:.2f}%",
            f"**Paddy pixels:** {paddy_pixels:,}",
            f"**Average crop height used:** {height_input:.2f} cm",
            f"**Rainfall used:** {rainfall_input:.2f} mm",
            f"**Average temperature used:** {temperature_input:.2f} °C",
            f"**Estimated yield:** {predicted_yield:.2f} MT/ha",
        ]

        if paddy_area_m2 is not None:
            summary_lines.insert(
                4,
                f"**Estimated physical paddy area:** {paddy_area_m2:.2f} m²"
            )

        st.info("\n\n".join(summary_lines))

    else:
        if uploaded_file is None:
            st.info("Upload a paddy field image to begin image analysis.")

    # --------------------------------------------------------
    # CURRENT MODEL SCOPE
    # --------------------------------------------------------

    st.markdown("""
    <div class="section">
        <div class="section-title">Current Model Scope</div>
        <div class="section-subtitle">What is currently available in the implemented system</div>
    </div>
    """, unsafe_allow_html=True)

    scope1, scope2 = st.columns(2)

    with scope1:
        st.markdown("""
        <div class="info-box">
        <b>XGBoost predictors:</b> The current yield prototype uses
        average crop height, rainfall, and average temperature because
        these are the variables available in the current yield dataset.
        </div>
        """, unsafe_allow_html=True)

    with scope2:
        st.markdown("""
        <div class="warning-box">
        <b>Important:</b> Paddy area and weed density are calculated by
        the segmentation system, but they are not paired with the current
        yield observations and therefore are not currently used as XGBoost
        predictors. Physical paddy area also requires a known image GSD.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# MODEL INFORMATION TAB
# ============================================================

with tabs[1]:

    st.markdown("## Model Information")

    st.markdown("""
    <div class="section">
        <div class="section-title">MobileNetV2-U-Net Segmentation</div>
        <div class="section-subtitle">
            Three-class semantic segmentation model
        </div>
    """, unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric("Classes", "3")
        st.write("Paddy • Weed • Background")

    with a2:
        st.metric("Input Size", "256 × 256")
        st.write("RGB image preprocessing")

    with a3:
        st.metric("Architecture", "MobileNetV2-U-Net")
        st.write("MobileNetV2 encoder with U-Net decoder")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
        <div class="section-title">Segmentation Class Mapping</div>
    """, unsafe_allow_html=True)

    st.dataframe(
        pd.DataFrame({
            "Model Class": [0, 1, 2],
            "Class": ["Paddy", "Background", "Weed"],
            "Dashboard Color": ["Yellow", "Green", "Red"]
        }),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
        <div class="section-title">XGBoost Yield Model</div>
    """, unsafe_allow_html=True)

    if dataset_loaded:
        st.write("Current predictors:")
        st.write("- Average crop height (cm)")
        st.write("- Rainfall (mm)")
        st.write("- Average temperature (°C)")
        st.write("Target: actual grain yield (MT/ha)")

        st.write("Model settings:")
        st.write("- XGBRegressor")
        st.write("- 300 estimators")
        st.write("- Maximum depth: 4")
        st.write("- Learning rate: 0.03")
        st.write("- Subsample: 0.8")
        st.write("- Column sampling: 0.8")
        st.write("- Objective: squared-error regression")
        st.write("- Random state: 42")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
        <div class="section-title">Spatial Validation</div>
    """, unsafe_allow_html=True)

    if dataset_loaded:
        st.write(
            f"Spatial groups used: {unique_groups}. "
            "GroupKFold keeps observations from the same spatial group together."
        )

        if len(cv_results_df) > 0:
            st.dataframe(
                cv_results_df.round(4),
                hide_index=True,
                use_container_width=True
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ABOUT TAB
# ============================================================

with tabs[2]:

    st.markdown("## About")

    st.markdown("""
    <div class="section">
        <div class="section-title">AI-Driven Paddy Field Monitoring</div>
        <p>
        The implemented system combines semantic image segmentation with
        an XGBoost regression prototype for crop-yield estimation.
        The segmentation model identifies paddy, weeds, and background
        at pixel level. The current yield model uses average crop height,
        rainfall, and average temperature to estimate grain yield in MT/ha.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
        <div class="section-title">Current Limitations</div>
        <ul>
            <li>The current XGBoost prototype uses three available predictors:
            crop height, rainfall, and average temperature.</li>
            <li>Paddy area and weed density are produced by segmentation but
            are not paired with the current yield observations.</li>
            <li>The current image dataset does not provide a reliable GSD,
            so physical area requires the actual image scale.</li>
            <li>Spatial validation is based on the currently available
            spatial groups.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
