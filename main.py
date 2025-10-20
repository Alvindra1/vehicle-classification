import numpy as np, pickle, os, cv2
import streamlit as st
from skimage.feature import hog

st.set_page_config(page_title='Vehicle Classification', layout= 'centered')
st.title("vehicle classification")

with open("xg_model.pkl",'rb') as file:
    model = pickle.load(file)

label = ["Bikes","Cars","Motorcycles","Planes","Ships","Trains"]

def hog_feature(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features= hog(gray_img,                           
                    orientations= 9,
                    pixels_per_cell = (8,8),
                    cells_per_block = (2,2),
                    block_norm = "L2-Hys",
                    transform_sqrt = True,
                    visualize=False
                   )
    return features

def orb_features(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=200)
    key, desc = orb.detectAndCompute(gray_img,None)
    if desc is None:
        desc = np.zeros((1,32))
    desc = desc.flatten()
    length = 6400
    if desc.shape[0] < length:
        desc = np.pad(desc, (0, length - desc.shape[0]))
    else:
        desc = desc[:length]
    return desc

def feature_extraction(img):
    hogs = hog_feature(img)
    orbs = orb_features(img)
    return np.concatenate([hogs,orbs])

upload_file = st.file_uploader("upload a vehicle image to be clasified",type=['jpg','jpeg','png'])

if upload_file is not None:
    bytes = np.frombuffer(upload_file.read(),np.uint8)
    img = cv2.imdecode(bytes, cv2.IMREAD_COLOR)

    if img is not None:
        st.info(f"processing image....")
        img = cv2.resize(img, (128,128))
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),use_container_width=True)
        if st.button("classify"):
            
            with st.spinner("loading..."):
                featured = feature_extraction(img).reshape(1,-1)

                pred = model.predict_proba(featured)[0]
                index = np.argmax(pred)
                pred_label = label[index]
                confidence = pred[index] * 100

            st.success(f"predicted class: {pred_label}")
            st.progress(float(confidence/100))
            st.write(f"confidence: {confidence:.2f}%")
    else:
        st.error("failed to read image")
else: 
    st.warning('please upload a vehicle image to be classify')




