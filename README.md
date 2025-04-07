
**Artwork Restoration Web App using Convolutional Autoencoder**

GitHub Link: https://github.com/vaishj2554/Art-restoration.git

**Structure of App**
PyTorch → Trained Model (.pth) → Streamlit (Python UI) → Local/Cloud Deployment

Artwork Restoration Using AI

**Aim:**
To restore damaged artwork images using a deep learning model trained on synthetically corrupted images. The project uses a Convolutional Autoencoder (CAE) to inpaint and reconstruct damaged regions of paintings or artwork.

**Dataset:**

Custom Art Restoration Dataset derived from WikiArt, featuring artwork images paired with synthetically generated damage like scribbles, blur, and patches.

**Model Summary:**

Architecture: Convolutional Autoencoder

Encoder: 3 Conv2D layers (downsampling from 128x128 to 16x16)

Decoder: 3 ConvTranspose2D layers (upsampling back to 128x128)

Activations: ReLU and Sigmoid

Normalization: BatchNorm used after each convolution

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Training Time (GPU - Tesla T4): 1 hour 2 minutes

Output: Model reconstructs high-quality images from corrupted inputs

**Deployment:**

Streamlit app for local/online use

Upload any clean artwork → model simulates damage → restores → shows side-by-side comparison

Real-time inference via GPU/CPU

**Web App Features:**

 1.Upload artwork (JPG/PNG)
 
 2.Damage is simulated realistically (scribbles, blur, patch)
 
 3.Restored image is shown side-by-side with original and damaged version
 
 4.No manual editing — fully automatic restoration

**Evaluation Metrics Implemented:**

MSE (Loss function)

Expert Visual Comparison (Streamlit side-by-side preview)

**Tools & Libraries:**

PyTorch (Deep Learning) , OpenCV (Damage simulation) , NumPy, Matplotlib, PIL (Preprocessing, visualization) ,Streamlit (Web app interface)
Trained on Kaggle (GPU - T4)

**Results:**

1.Visual accuracy of restored images is high — fine texture and structure preserved

2.Scribbles and patches removed well from test samples

3.Model generalizes well to unseen art styles



