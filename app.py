import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
from backend import *
# Load the model (assuming the model, encoder, and decoder are defined and saved)
model = torch.load('./models/model_1000.pkl', map_location=torch.device('cpu'))
model.eval()

# Title for Streamlit app
st.title("Multi-Image Steganography")

# File uploaders for cover and secret images
cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg"])
secret_image_1_file = st.file_uploader("Upload Secret Image 1", type=["png", "jpg", "jpeg"])
secret_image_2_file = st.file_uploader("Upload Secret Image 2", type=["png", "jpg", "jpeg"])
secret_image_3_file = st.file_uploader("Upload Secret Image 3", type=["png", "jpg", "jpeg"])

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Adjust dimensions as per model input
    transforms.ToTensor(),
])

# Process and display results if all images are uploaded
if cover_image_file and secret_image_1_file and secret_image_2_file and secret_image_3_file:
    # Load and transform the images
    cover_image = transform(Image.open(cover_image_file)).unsqueeze(0)
    secret_image_1 = transform(Image.open(secret_image_1_file)).unsqueeze(0)
    secret_image_2 = transform(Image.open(secret_image_2_file)).unsqueeze(0)
    secret_image_3 = transform(Image.open(secret_image_3_file)).unsqueeze(0)

    # Move images to CPU (or GPU if available)
    device = torch.device("cpu")  # Change to "cuda" if using GPU
    cover_image, secret_image_1, secret_image_2, secret_image_3 = (
        cover_image.to(device),
        secret_image_1.to(device),
        secret_image_2.to(device),
        secret_image_3.to(device)
    )

    # Perform encoding and decoding with the model
    with torch.no_grad():
        hidden_image, reveal_image_1, reveal_image_2, reveal_image_3 = model(
            cover_image,
            secret_image_1,
            secret_image_2,
            secret_image_3,
            None,  # None for hidden image input in 'full' mode
            mode='full'
        )

    # Convert tensors back to images for display
    cover_image_pil = TF.to_pil_image(cover_image.squeeze().cpu())
    secret_image_1_pil = TF.to_pil_image(secret_image_1.squeeze().cpu())
    secret_image_2_pil = TF.to_pil_image(secret_image_2.squeeze().cpu())
    secret_image_3_pil = TF.to_pil_image(secret_image_3.squeeze().cpu())
    
    hidden_image_pil = TF.to_pil_image(hidden_image.squeeze().cpu())
    reveal_image_1_pil = TF.to_pil_image(reveal_image_1.squeeze().cpu())
    reveal_image_2_pil = TF.to_pil_image(reveal_image_2.squeeze().cpu())
    reveal_image_3_pil = TF.to_pil_image(reveal_image_3.squeeze().cpu())

    # Display the images in two rows
    st.subheader("Input Images (First Row)")
    st.image([cover_image_pil, secret_image_1_pil, secret_image_2_pil, secret_image_3_pil],
             caption=["Cover Image", "Secret Image 1", "Secret Image 2", "Secret Image 3"],
             width=150)

    st.subheader("Output Images (Second Row)")
    st.image([hidden_image_pil, reveal_image_1_pil, reveal_image_2_pil, reveal_image_3_pil],
             caption=["Hidden (Encoded) Image", "Revealed Image 1", "Revealed Image 2", "Revealed Image 3"],
             width=150)

else:
    st.write("Please upload all four images to proceed.")
