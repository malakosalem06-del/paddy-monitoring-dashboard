import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision import transforms
from PIL import Image
import numpy as np


st.set_page_config(
    page_title="Paddy Field Monitoring",
    layout="wide"
)


st.title("AI-Driven Paddy Field Monitoring")
st.write("AI-based semantic segmentation of paddy, weeds, and background.")


st.subheader("Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Paddy IoU", "53.96%")

with col2:
    st.metric("Weed IoU", "58.69%")

with col3:
    st.metric("Background IoU", "89.01%")

with col4:
    st.metric("Pixel Accuracy", "89.85%")

with col5:
    st.metric("Paddy Recall", "86.90%")


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


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
        "The AI model could not be loaded."
    )

    st.code(str(e))

    st.stop()


st.subheader("Upload Image")

uploaded_file = st.file_uploader(
    "Upload a paddy field image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    input_image = image.resize(
        (256, 256)
    )


    image_tensor = transforms.ToTensor()(
        input_image
    )


    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)


    with torch.no_grad():

        output = model(
            image_tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        )


    pred_mask = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
    )


    mask_resized = Image.fromarray(
        pred_mask.astype(np.uint8)
    ).resize(
        image.size,
        resample=Image.Resampling.NEAREST
    )


    mask_array = np.array(
        mask_resized
    )


    paddy_mask = (
        mask_array == 0
    )

    background_mask = (
        mask_array == 1
    )

    weed_mask = (
        mask_array == 2
    )


    paddy_percentage = (
        paddy_mask.mean() * 100
    )

    background_percentage = (
        background_mask.mean() * 100
    )

    weed_percentage = (
        weed_mask.mean() * 100
    )


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


    st.subheader(
        "Segmentation Summary"
    )


    st.write(
        f"""
The AI model classified the uploaded image into three
classes: Paddy, Weed, and Background.

Paddy: {paddy_percentage:.2f}%

Weed: {weed_percentage:.2f}%

Background: {background_percentage:.2f}%
"""
    )
