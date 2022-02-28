import streamlit as st
import helper
import numpy as np
import torch 
from torchvision import models, transforms
from PIL import Image 


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# function to show image and its predicted breed
def show_pet_and_breed(tags, image):
    """
    Shows an image of a pet and prints out its predicted breed and probability using the tags dictionary
    """
    breed = tags['label'] # the predicted breed
    pet_category = 'cat' if breed[0].isupper() else 'dog' # capitalized breed categories are cats, otherwise dogs
    breed = breed.lower()
    breed = ' '.join(breed.split('_')) # multi-word categories are joined using '_', so replace it with space
    article = 'an' if breed[0] in ['a', 'e', 'i', 'o', 'u'] else 'a' # the definite article for category to be printed!
    st.image(image, caption="I am {} percent sure this is {} {} {}".format(round(tags['prob']*100), article, breed , pet_category))


def get_image(query):
    image = Image.open(query)
    img_tensor = preprocess(image).unsqueeze(0)
    return img_tensor, image


# Classifier model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 37) # 37 is the number of categories, i.e., breeds
model.load_state_dict(torch.load('files/best_model.pth', map_location=device))
model = model.to(device)
model.eval()


# UI layout
st.set_page_config(page_title="Pet Breed Classification")
st.markdown(
        body=helper.UI.css,
        unsafe_allow_html=True,
)
# Sidebar
st.sidebar.markdown(helper.UI.about_block, unsafe_allow_html=True)

# Title
st.header("Pet Breed Classification")

# File uploader
upload_cell, preview_cell = st.columns([12, 1])
query = upload_cell.file_uploader("")

# If file is uploaded
if query:
    # if clicked on 'classify' button
    if st.button(label="Classify"):
        # read image and process it for the model
        img_tensor, image = get_image(query)
        with torch.no_grad():
            output = torch.nn.Softmax(dim=1)(model(img_tensor.to(device)))
        prob, label = torch.max(output, 1)
        # show the image and its predicted greed!
        prob = prob.cpu().squeeze().tolist()
        label = label.cpu().squeeze().tolist()

        tags = {"prob": prob, "label": helper.classToBreed[label]}
        show_pet_and_breed(tags, image)