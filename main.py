import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

dic = {0: 'dog', 1: 'cat'}

model = tf.keras.models.load_model('model.h5')

def predict_label(img):
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    p = model.predict(img)
    predicted_class = np.argmax(p, axis=1)[0]
    return dic[predicted_class]

# Streamlit app
def main():
    st.title('Image Classifier')
    st.write('Upload an image for prediction')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        if st.button('Predict'):
            label = predict_label(img)
            st.write(f'Prediction: {label}')

if __name__ == '__main__':
    main()
