import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

model_dir = os.path.join(os.getcwd(), "models")
print(model_dir)
# Load your models
models = {
    "Model Xception": load_model(os.path.join(model_dir, "xception_model.h5")),
    "Model MobileNet": load_model(os.path.join(model_dir, "mobilenetV2_model.h5")),
    "Model ResNet": load_model(os.path.join(model_dir, "resnet50_model.h5")),
    "Model EfficientNet": load_model(os.path.join(model_dir, "efficientnet_model.h5")),
}

# Dictionary to map model names to their last conv layer names
last_conv_layer_names = {
    "Model Xception": "block14_sepconv2_act",
    "Model MobileNet": "out_relu",  # out_relu
    "Model ResNet": "conv5_block3_out",
    # "Model EfficientNet": load_model("efficientnet_model.h5"),
}


# Function to preprocess the image
def preprocess_image(image):
    size = (240, 240)  # Change size based on your model input
    image = ImageOps.fit(image, size)
    image = np.asarray(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def predict_image_class(image, model, class_names, is_binary=False):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)

    if is_binary:
        predicted_class = class_names[int(prediction[0] > 0.5)]
        prediction_probs = {
            class_names[0]: float(1 - prediction[0]),
            class_names[1]: float(prediction[0]),
        }
    else:
        predicted_class = class_names[np.argmax(prediction)]
        prediction_probs = {
            class_names[i]: float(prediction[0][i]) for i in range(len(class_names))
        }

    return predicted_class, prediction_probs


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, pred_index=None, is_binary=False
):
    # Create a model that maps the input image to the activations
    # of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0]) if not is_binary else (preds[0] > 0.5)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by "how important this channel is" regarding the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def display_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = np.array(jet_heatmap)

    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    return superimposed_img


# Streamlit app
st.title("Ant Species Image Classification")

selected_model_name = st.selectbox("Select a model:", list(models.keys()))
selected_model = models[selected_model_name]
last_conv_layer_name = last_conv_layer_names[selected_model_name]

uploaded_files = st.file_uploader(
    "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)
if uploaded_files:
    class_names = [
        "Argentine-ants",
        "Black-crazy-ants",
        "fire-ants",
        "Trap-jaw-ants",
        "Weaver-ants",
        "Yellow-crazy-ants",
    ]

    # Determine if the model is binary
    is_binary = selected_model_name == "Model ResNet"

    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption=f"Uploaded Image: {uploaded_file.name}",
                use_column_width=True,
            )
            st.write("Classifying...")

            predicted_class, prediction_probs = predict_image_class(
                image, selected_model, class_names, is_binary
            )

            st.write(f"Prediction: {predicted_class}")
            st.write("Prediction Probabilities:")
            st.write(prediction_probs)

            # Only generate heatmap for "Model CNN"
            # if selected_model_name == "Model CNN":
            # Generate heatmap
            img_array = preprocess_image(image)
            heatmap = make_gradcam_heatmap(
                img_array,
                selected_model,
                last_conv_layer_name,  # is_binary=is_binary
            )

            # Display heatmap
            superimposed_img = display_gradcam(np.array(image), heatmap)
            st.image(
                superimposed_img, caption="Grad-CAM Heatmap", use_column_width=True
            )
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
