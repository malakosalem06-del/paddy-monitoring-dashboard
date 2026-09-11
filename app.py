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


st.set_page_config(
    page_title="Paddy Field Monitoring",
    layout="wide"
)

st.title("AI-Driven Paddy Field Monitoring")
st.write(
    "AI-based semantic segmentation of paddy, weeds, and background "
    "with crop yield estimation."
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
            1280, 320, 2, stride=2
        )

        self.conv1 = nn.Conv2d(
            384, 320, 3, padding=1
        )

        self.up2 = nn.ConvTranspose2d(
            320, 64, 2, stride=2
        )

        self.conv2 = nn.Conv2d(
            96, 64, 3, padding=1
        )

        self.up3 = nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )

        self.conv3 = nn.Conv2d(
            56, 32, 3, padding=1
        )

        self.up4 = nn.ConvTranspose2d(
            32, 16, 2, stride=2
        )

        self.conv4 = nn.Conv2d(
            40, 16, 3, padding=1
        )

        self.up5 = nn.ConvTranspose2d(
            16, 16, 2, stride=2
        )

        self.final = nn.Conv2d(
            16, 3, 1
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
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD SEGMENTATION MODEL
# ============================================================

model = MobileNetV2UNet().to(device)

model_path = "best_mobilenetv2_unet_improved_3class.pth"


try:

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

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

    st.success(
        "AI segmentation model loaded successfully."
    )

except Exception as e:

    st.error(
        "The AI segmentation model could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("Segmentation Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Paddy IoU",
        "53.96%"
    )

with col2:
    st.metric(
        "Weed IoU",
        "58.69%"
    )

with col3:
    st.metric(
        "Background IoU",
        "89.01%"
    )

with col4:
    st.metric(
        "Pixel Accuracy",
        "89.85%"
    )

with col5:
    st.metric(
        "Paddy Recall",
        "86.90%"
    )


# ============================================================
# LOAD XGBOOST DATASET
# ============================================================

st.subheader("Yield Prediction Model")

data_path = "rice_xgboost_training_data_combined.csv"


try:

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    yield_data = pd.read_csv(data_path)

except Exception as e:

    st.error(
        "The XGBoost training dataset could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# PREPARE XGBOOST DATA
# ============================================================

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


if len(missing_columns) > 0:

    st.error(
        "The yield dataset is missing required columns."
    )

    st.write(missing_columns)

    st.stop()


xgboost_data = yield_data[
    required_columns
].copy()


xgboost_data = xgboost_data.dropna()


X = xgboost_data[features]

y = xgboost_data[target]

groups = xgboost_data["spatial_group"]


# ============================================================
# TRAIN XGBOOST MODEL
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
# SPATIAL CROSS VALIDATION
# ============================================================

unique_groups = groups.nunique()

cv_results = []

if unique_groups >= 2:

    n_splits = unique_groups

    group_kfold = GroupKFold(
        n_splits=n_splits
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


# ============================================================
# DISPLAY XGBOOST DATASET INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Training Observations",
        len(xgboost_data)
    )

with col2:
    st.metric(
        "Spatial Groups",
        unique_groups
    )

with col3:
    st.metric(
        "Yield Range",
        f"{y.min():.2f} - {y.max():.2f} MT/ha"
    )


# ============================================================
# DISPLAY SPATIAL VALIDATION
# ============================================================

st.subheader("Spatial Validation")

if len(cv_results_df) > 0:

    mean_mae = cv_results_df["MAE"].mean()

    mean_rmse = cv_results_df["RMSE"].mean()

    valid_r2 = cv_results_df["R2"].dropna()

    if len(valid_r2) > 0:
        mean_r2 = valid_r2.mean()
    else:
        mean_r2 = np.nan


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Mean MAE",
            f"{mean_mae:.3f} MT/ha"
        )

    with col2:
        st.metric(
            "Mean RMSE",
            f"{mean_rmse:.3f} MT/ha"
        )

    with col3:

        if np.isnan(mean_r2):
            st.metric(
                "Mean R²",
                "N/A"
            )

        else:
            st.metric(
                "Mean R²",
                f"{mean_r2:.3f}"
            )


    st.write(
        "Spatial validation uses the available locations as "
        "groups so that observations from the same location "
        "are not mixed between training and validation."
    )


else:

    st.warning(
        "Spatial validation could not be performed."
    )


# ============================================================
# DISPLAY TRAINING LOCATIONS
# ============================================================

st.write("Locations used for spatial validation:")

locations = sorted(
    xgboost_data["spatial_group"]
    .astype(str)
    .unique()
)

st.write(
    ", ".join(locations)
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("Upload Paddy Field Image")

uploaded_file = st.file_uploader(
    "Upload a paddy field image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ========================================================
    # SEGMENTATION
    # ========================================================

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


    # ========================================================
    # CLASS MASKS
    # ========================================================

    paddy_mask = mask_array == 0

    background_mask = mask_array == 1

    weed_mask = mask_array == 2


    # ========================================================
    # PERCENTAGES
    # ========================================================

    paddy_percentage = (
        paddy_mask.mean() * 100
    )

    background_percentage = (
        background_mask.mean() * 100
    )

    weed_percentage = (
        weed_mask.mean() * 100
    )


    # ========================================================
    # PIXEL COUNTS
    # ========================================================

    total_pixels = mask_array.size

    paddy_pixels = int(
        paddy_mask.sum()
    )

    weed_pixels = int(
        weed_mask.sum()
    )

    background_pixels = int(
        background_mask.sum()
    )


    # ========================================================
    # SEGMENTATION IMAGE
    # ========================================================

    original_array = np.array(
        image
    )

    segmentation = np.zeros_like(
        original_array
    )


    segmentation[paddy_mask] = [
        255,
        255,
        0
    ]

    segmentation[background_mask] = [
        0,
        0,
        0
    ]

    segmentation[weed_mask] = [
        255,
        0,
        0
    ]


    # ========================================================
    # OVERLAY
    # ========================================================

    overlay = original_array.copy()

    alpha = 0.45


    overlay[paddy_mask] = (
        alpha * segmentation[paddy_mask]
        +
        (1 - alpha) * overlay[paddy_mask]
    ).astype(np.uint8)


    overlay[weed_mask] = (
        alpha * segmentation[weed_mask]
        +
        (1 - alpha) * overlay[weed_mask]
    ).astype(np.uint8)


    # ========================================================
    # DISPLAY IMAGES
    # ========================================================

    st.subheader(
        "AI Segmentation Results"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )


    with col2:

        st.image(
            segmentation,
            caption="AI Prediction",
            use_container_width=True
        )


    with col3:

        st.image(
            overlay,
            caption="Segmentation Overlay",
            use_container_width=True
        )


    # ========================================================
    # FIELD ANALYSIS
    # ========================================================

    st.subheader(
        "Field Analysis"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Paddy Area",
            f"{paddy_percentage:.2f}%"
        )


    with col2:

        st.metric(
            "Weed Area",
            f"{weed_percentage:.2f}%"
        )


    with col3:

        st.metric(
            "Background",
            f"{background_percentage:.2f}%"
        )


    # ========================================================
    # PIXEL INFORMATION
    # ========================================================

    st.subheader(
        "Segmentation Pixel Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Paddy Pixels",
            f"{paddy_pixels:,}"
        )


    with col2:

        st.metric(
            "Weed Pixels",
            f"{weed_pixels:,}"
        )


    with col3:

        st.metric(
            "Background Pixels",
            f"{background_pixels:,}"
        )


    # ========================================================
    # PHYSICAL AREA
    # ========================================================

    st.subheader(
        "Physical Paddy Area"
    )

    st.write(
        "Enter the Ground Sampling Distance (GSD) of the image "
        "to calculate physical paddy area. Do not enter an "
        "estimated value unless the image scale is known."
    )


    gsd_cm = st.number_input(
        "GSD (cm per pixel)",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.01
    )


    gsd_m = gsd_cm / 100.0


    pixel_area_m2 = (
        gsd_m ** 2
    )


    paddy_area_m2 = (
        paddy_pixels * pixel_area_m2
    )


    st.metric(
        "Estimated Paddy Area",
        f"{paddy_area_m2:.2f} m²"
    )


    # ========================================================
    # YIELD INPUTS
    # ========================================================

    st.subheader(
        "Yield Prediction Inputs"
    )

    st.write(
        "The current yield dataset provides crop height, "
        "rainfall, and temperature. These three variables "
        "are used by the current XGBoost prototype."
    )


    mean_height = float(
        xgboost_data["avg_height_cm"].mean()
    )

    mean_rainfall = float(
        xgboost_data["rainfall_mm"].mean()
    )

    mean_temperature = float(
        xgboost_data["avg_temp_c"].mean()
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        height_input = st.number_input(
            "Average Crop Height (cm)",
            min_value=0.0,
            max_value=300.0,
            value=round(
                mean_height,
                2
            ),
            step=0.1
        )


    with col2:

        rainfall_input = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=5000.0,
            value=round(
                mean_rainfall,
                2
            ),
            step=1.0
        )


    with col3:

        temperature_input = st.number_input(
            "Average Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=round(
                mean_temperature,
                2
            ),
            step=0.1
        )


    # ========================================================
    # XGBOOST PREDICTION
    # ========================================================

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


    # ========================================================
    # YIELD RESULT
    # ========================================================

    st.subheader(
        "Estimated Yield"
    )


    st.metric(
        "Predicted Yield",
        f"{predicted_yield:.2f} MT/ha"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.subheader(
        "Field Monitoring Summary"
    )


    st.write(
        f"""
The AI segmentation model classified the uploaded image
into Paddy, Weed, and Background.

Paddy coverage: {paddy_percentage:.2f}%

Weed coverage: {weed_percentage:.2f}%

Background coverage: {background_percentage:.2f}%

Paddy pixels: {paddy_pixels:,}

Estimated physical paddy area: {paddy_area_m2:.2f} m²

Average crop height used for yield prediction:
{height_input:.2f} cm

Rainfall used for yield prediction:
{rainfall_input:.2f} mm

Average temperature used for yield prediction:
{temperature_input:.2f} °C

Estimated yield:
{predicted_yield:.2f} MT/ha
"""
    )


# ============================================================
# PROJECT LIMITATION
# ============================================================

st.subheader(
    "Current Model Scope"
)

st.write(
    "The current XGBoost prototype uses crop height, rainfall, "
    "and average temperature because these are the variables "
    "available with the current yield dataset. Paddy area and "
    "weed density are calculated by the segmentation system, "
    "but they are not yet available as paired training features "
    "for the yield dataset. Therefore, they are not currently "
    "used as XGBoost predictors."
)

st.write(
    "The physical paddy area calculation requires a known GSD. "
    "The current ground-based image dataset does not provide a "
    "reliable GSD value, so the displayed area should only be "
    "treated as physically meaningful when the user supplies "
    "the actual image scale."
)
