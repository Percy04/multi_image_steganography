import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
from io import BytesIO
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
    transforms.Resize((1000, 1000)),  # Adjust dimensions as per model input
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

    # Convert tensors back to images for display and download
    hidden_image_pil = TF.to_pil_image(hidden_image.squeeze().cpu())
    reveal_image_1_pil = TF.to_pil_image(reveal_image_1.squeeze().cpu())
    reveal_image_2_pil = TF.to_pil_image(reveal_image_2.squeeze().cpu())
    reveal_image_3_pil = TF.to_pil_image(reveal_image_3.squeeze().cpu())

    # Display the images in two rows
    st.subheader("Input Images (First Row)")
    st.image([cover_image_file, secret_image_1_file, secret_image_2_file, secret_image_3_file],
             caption=["Cover Image", "Secret Image 1", "Secret Image 2", "Secret Image 3"],
             width=150)

    st.subheader("Output Images (Second Row)")
    st.image([hidden_image_pil, reveal_image_1_pil, reveal_image_2_pil, reveal_image_3_pil],
             caption=["Hidden (Encoded) Image", "Revealed Image 1", "Revealed Image 2", "Revealed Image 3"],
             width=150)

    # Prepare images for download by converting them to byte streams
    def convert_to_bytes(img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    hidden_image_bytes = convert_to_bytes(hidden_image_pil)
    reveal_image_1_bytes = convert_to_bytes(reveal_image_1_pil)
    reveal_image_2_bytes = convert_to_bytes(reveal_image_2_pil)
    reveal_image_3_bytes = convert_to_bytes(reveal_image_3_pil)

    # Create download buttons for each output image
    st.subheader("Download Output Images")
    st.download_button("Download Hidden (Encoded) Image", data=hidden_image_bytes, file_name="hidden_image.png", mime="image/png")
    st.download_button("Download Revealed Image 1", data=reveal_image_1_bytes, file_name="reveal_image_1.png", mime="image/png")
    st.download_button("Download Revealed Image 2", data=reveal_image_2_bytes, file_name="reveal_image_2.png", mime="image/png")
    st.download_button("Download Revealed Image 3", data=reveal_image_3_bytes, file_name="reveal_image_3.png", mime="image/png")
else:
    st.write("Please upload all four images to proceed.")
