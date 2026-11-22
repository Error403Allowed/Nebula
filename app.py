import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# page setup
st.set_page_config(
    page_title="Cosmology Image Classifier",
    page_icon="✨",
    layout="centered"
)

# make sure these match the folder names inside dataset/train exactly!
CLASS_NAMES = ['galaxy', 'nebula', 'star', 'supernova']

# standard transforms for resnet (resize, tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the trained model and cache it so it doesn't reload every time the user clicks
@st.cache_resource
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # build the model structure
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    # load our saved weights file
    model_path = 'cosmology_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.warning("Hold up! 'cosmology_model.pth' not found. Make sure to run train.py first.")
        
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ui stuff
st.title("Cosmology Image Classifier")
st.write("Drop a space image below to see what the model thinks it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # show the uploaded pic
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify Image'):
        with st.spinner('Running inference...'):
            # prep the image for the model
            tensor_image = transform(image)
            input_tensor = tensor_image.unsqueeze(0).to(device)  # type: ignore
            
            # get predictions
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, preds = torch.max(probabilities, 0)
                
            # wrap preds.item() in int() to satisfy Pylance list indexing
            predicted_class = CLASS_NAMES[int(preds.item())]
            confidence_score = conf.item() * 100
            
            # display main result
            st.success(f"**Prediction:** {predicted_class.upper()} ({confidence_score:.2f}% confidence)")
            
            # breakdown for all classes
            st.subheader("Confidence Breakdown:")
            for i, class_name in enumerate(CLASS_NAMES):
                score = probabilities[i].item()
                st.progress(score, text=f"{class_name}: {score * 100:.2f}%")
