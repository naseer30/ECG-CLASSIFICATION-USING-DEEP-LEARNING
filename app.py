
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
from keras.models import load_model
from Ecg import divide_leads, preprocess_lead
import cv2


# Configure Streamlit to disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)

class_names = ['History_of_MI', 'Myocardial Infarction', 'Normal beat', 'Abnormal Heartbeat']
# Load the trained CNN model
model = load_model('./my_cnn_model_1.h5')

# Function to preprocess a single image and save leads
def preprocess_and_save_single(uploaded_file, output_folder):
    # Save the uploaded file to a temporary location
    image_path = os.path.join(output_folder, uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.read())

    # Load the ECG image
    image = np.array(Image.open(image_path))

    # Convert to grayscale if image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        grayscale_image = color.rgb2gray(image)
    else:
        grayscale_image = image

    # Divide the ECG image into leads
    leads = divide_leads(grayscale_image)

    # Create a folder for the ECG image
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_folder = os.path.join(output_folder, image_name)
    os.makedirs(image_folder, exist_ok=True)

    # Preprocess each lead and save
    preprocessed_leads = []
    for i, lead in enumerate(leads, start=1):
        preprocessed_lead = preprocess_lead(lead)
        lead_output_path = os.path.join(image_folder, f'{image_name}_lead_{i}.png')  
        preprocessed_lead.save(lead_output_path)
        preprocessed_leads.append(preprocessed_lead)

    # Remove the temporary image file
    os.remove(image_path)

    return leads, preprocessed_leads, grayscale_image

# Function to preprocess a single lead image and resize it
def preprocess_and_resize_lead(lead_image):
    try:
        # Convert to numpy array if not already
        lead_array = np.array(lead_image)
        # Check if image is grayscale
        if len(lead_array.shape) == 2:
            # Convert to RGB
            lead_array = cv2.cvtColor(lead_array, cv2.COLOR_GRAY2RGB)
        # Resize the image
        resized_image = cv2.resize(lead_array, (150, 150))
        return resized_image
    except Exception as e:
        st.error(f"Error preprocessing and resizing lead: {e}")
        # Return a placeholder image or handle the error appropriately
        return np.zeros((150, 150, 3))  # Placeholder image with zeros



# Main function
def main():

    st.markdown("<h1 style='text-align: center;'>ECG Image Classification</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("Uploaded ECG Image")
        st.image(uploaded_file, caption="Uploaded ECG Image", use_column_width=True)
        
        # Preprocess and save leads only if an image is uploaded
        temp_folder = './temp'
        original_leads, preprocessed_leads, grayscale_image = preprocess_and_save_single(uploaded_file, temp_folder)
        if preprocessed_leads is not None:
            # Display grayscale image
            with st.expander("Grayscale Image"):
                plt.imshow(grayscale_image, cmap='gray')
                plt.axis('off')
                st.pyplot()

            # Display divided leads of grayscale image
            with st.expander("Divided Leads of Grayscale Image"):
                fig, axs = plt.subplots(4, 3, figsize=(10, 10))
                fig.subplots_adjust(hspace=0.5)
                for i, lead in enumerate(original_leads):
                    if i < 12:
                        axs[i // 3, i % 3].imshow(lead, cmap='gray')
                        axs[i // 3, i % 3].axis('off')
                        axs[i // 3, i % 3].set_title(f"Lead {i+1}")
                st.pyplot(fig)

            # Display divided 13th lead separately
            with st.expander("Divided 13th Lead"):
                plt.figure(figsize=(10, 10))
                plt.imshow(original_leads[-1], cmap='gray')
                plt.axis('off')
                st.pyplot()

            # Display preprocessed 12 leads
            with st.expander("Preprocessed 12 Leads"):
                fig_preprocessed, axs_preprocessed = plt.subplots(4, 3, figsize=(10, 10))
                fig_preprocessed.subplots_adjust(hspace=0.5)
                for i, lead in enumerate(preprocessed_leads[:12]):
                    axs_preprocessed[i // 3, i % 3].imshow(lead, cmap='gray')
                    axs_preprocessed[i // 3, i % 3].axis('off')
                    axs_preprocessed[i // 3, i % 3].set_title(f"Lead {i+1}")
                st.pyplot(fig_preprocessed)

            # Display preprocessed 13th lead
            with st.expander("Preprocessed 13th Lead"):
                plt.figure(figsize=(10, 10))
                plt.imshow(preprocessed_leads[-1], cmap='gray')
                plt.axis('off')
                st.pyplot()

            # Resize preprocessed leads to (150, 150)
            resized_leads = [preprocess_and_resize_lead(lead) for lead in preprocessed_leads]
            resized_leads = np.array(resized_leads)
            # Make predictions using the model
            predictions = model.predict(resized_leads)
            # Get the predicted class indices for each prediction
            predicted_classes = np.argmax(predictions, axis=1)
            # Display the predictions
            with st.expander("Prediction"):
                for i, prediction in enumerate(predictions):
                    # Get the predicted class index
                    predicted_class_index = np.argmax(prediction)
                    # Get the predicted class name
                    predicted_class_name = class_names[predicted_class_index]
                    # Display the prediction along with the class name
                    st.write(f"Lead {i+1} prediction: {prediction}")
                flattened_predictions = np.sum(predictions, axis=0)
                # Determine the overall predicted class index
                overall_prediction = np.argmax(flattened_predictions)
                # Get the overall predicted class name
                overall_class = class_names[overall_prediction]
                # Display the overall predicted class name
                st.markdown(f"<h4>uploaded image belongs to: {overall_class}</h4>", unsafe_allow_html=True)                # Load the history object
                history = np.load('train_history_1.npy', allow_pickle=True).item()

                # Access accuracy and loss values
                accuracy = history.get('accuracy', None)
                loss = history.get('loss', None)
                val_accuracy = history.get('val_accuracy', None)
                val_loss = history.get('val_loss', None)
            with st.expander("Accuracy and Loss"):
                 # Display accuracy and loss in percentage format
                if accuracy is not None:
                    st.markdown(f"<h5>Accuracy: {accuracy[-1] * 100:.2f}%</h5>", unsafe_allow_html=True)
                if val_accuracy is not None:
                    st.markdown(f"<h5>Validation Accuracy: {val_accuracy[-1] * 100:.2f}%</h5>", unsafe_allow_html=True)
                if loss is not None:
                    st.markdown(f"<h5>Loss: {loss[-1]}</h5>", unsafe_allow_html=True)
                if val_loss is not None:
                    st.markdown(f"<h5>Validation Loss: {val_loss[-1]}</h5>", unsafe_allow_html=True)
            

if __name__ == "__main__":
     main()
