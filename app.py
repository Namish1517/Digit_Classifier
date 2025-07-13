import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_cnn.h5")

model = load_model()


def init_canvas_state():
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0

init_canvas_state()


st.set_page_config(page_title="Digit Classifier", layout="centered")
st.title("Handwritten Digit Classifier")


st.markdown("Draw a digit (0-9) below:")
canvas_result = st_canvas(
    fill_color="white", 
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}"
)


col1, col2 = st.columns(2)
with col1:
    if st.button("Clear Canvas"):
       st.session_state.canvas_key += 1 
       st.rerun()


with col2:
    predict = st.button("Predict")


def preprocess_image(raw_image):
    img = raw_image.convert("L")                     
    img = img.resize((28, 28))                    
    img = ImageOps.invert(img)                       
    img = np.array(img).astype("float32") / 255.0   
    img = img.reshape((1, 28, 28, 1))                
    return img


if predict:
    if canvas_result.image_data is not None:
        
        drawn = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
        processed = preprocess_image(drawn)
        probs = model.predict(processed)[0]
        digit = int(np.argmax(probs))
        confidence = float(np.max(probs))

        st.subheader(f"Prediction: {digit}")
        st.write(f"Confidence: {confidence:.2f}")
        st.bar_chart(probs)
    else:
        st.warning("Please draw a digit before predicting.")
