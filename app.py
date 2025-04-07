import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# --- Define your model architecture ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- Load model ---
@st.cache_resource
def load_model():
    model = Autoencoder()
    model.load_state_dict(torch.load("realistic_art_restoration.pth", map_location="cpu"))
    model.eval()
    return model

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def preprocess_image(uploaded_img):
    img = Image.open(uploaded_img).convert("RGB")
    img_resized = transform(img)
    return img, img_resized.unsqueeze(0)  # return PIL and tensor with batch dim

# --- Inference ---
def restore_image(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    return output.squeeze(0).permute(1, 2, 0).numpy()

# --- Streamlit App ---
st.title("🖼️ AI Art Restoration")
st.markdown("Upload a damaged artwork image. The model will try to restore it!")

uploaded_file = st.file_uploader("Upload Damaged Artwork", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load model
    model = load_model()

    # Preprocess
    original_pil, input_tensor = preprocess_image(uploaded_file)

    # Restore
    restored_np = restore_image(model, input_tensor)

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_pil, caption="Damaged Image", use_column_width=True)
    with col2:
        st.image(restored_np, caption="Restored Image", use_column_width=True)

    st.success("✅ Restoration complete!")