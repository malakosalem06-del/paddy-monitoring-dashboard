import streamlit as st

st.set_page_config(
    page_title="Paddy Field Monitoring",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AI-Driven Paddy Field Monitoring")
st.write("Upload a UAV image to analyze paddy field vegetation.")

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image
import numpy as np

class MobileNetV2UNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        self.encoder = backbone.features

        self.up1 = nn.ConvTranspose2d(1280, 320, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(320, 96, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(96, 32, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.up5 = nn.ConvTranspose2d(16, 16, 2, stride=2)

        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x = self.encoder(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)

        x = self.final(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MobileNetV2UNet().to(device)

model.load_state_dict(
    torch.load(
        "best_mobilenetv2_unet.pth",
        map_location=device
    )
)

model.eval()

uploaded_file = st.file_uploader(
    "Upload a UAV image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    input_image = image.resize((256, 256))

    image_tensor = transforms.ToTensor()(input_image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)
        prediction = (prediction > 0.5).float()

    pred_mask = prediction.squeeze().cpu().numpy()

    original_array = np.array(image)

    mask_resized = Image.fromarray(
        (pred_mask * 255).astype(np.uint8)
    ).resize(image.size)

    mask_array = np.array(mask_resized) > 127

    overlay = original_array.copy()
    overlay[mask_array] = (
        0.7 * overlay[mask_array] +
        0.3 * np.array([255, 0, 0])
    ).astype(np.uint8)

    weedy_percentage = mask_array.mean() * 100
    background_percentage = 100 - weedy_percentage

    st.subheader("AI Segmentation Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="Original Image")

   with col2:
    mask_display = (mask_array * 255).astype(np.uint8)
    st.image(mask_display, caption="AI Prediction")
       
    with col3:
        st.image(overlay, caption="Segmentation Overlay")

    st.metric("Weedy Rice", f"{weedy_percentage:.2f}%")
    st.metric("Background", f"{background_percentage:.2f}%")
