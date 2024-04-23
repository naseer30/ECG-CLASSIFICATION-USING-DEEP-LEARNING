import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import streamlit as st
import pandas as pd

def evaluate_model():
    # Define directories for validation data
    validation_data_dir = './PreprocessedData/validation'

    # Configure data generator for validation data
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Set image dimensions
    target_size = (150, 150)

    # Define the generator for validation data
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=target_size,
        batch_size=64,
        class_mode='categorical',  # Use 'categorical' for multi-class classification
        shuffle=False
    )
    
    # Get class names from the generator
    class_names = list(validation_generator.class_indices.keys())

    # Load the pre-trained model
    model = load_model('my_cnn_model_1.h5')
    if model is None:
        st.error("Error: Failed to load the model.")
        return
    
    # Evaluate the model
    validation_loss, validation_accuracy = model.evaluate(validation_generator)
    st.write("Validation Accuracy:", validation_accuracy)
    st.write("Validation Loss:", validation_loss)
    

    # Predict classes for validation data
    y_true = validation_generator.classes  # True labels
    y_pred = np.argmax(model.predict(validation_generator), axis=-1)  # Predicted labels

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    st.write("Confusion Matrix:")
    st.table(cm_df)
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Print the results
    st.write("Total Precision:", precision)
    st.write("Total Recall:", recall)
    st.write("Total F1-score:", f1)
    # Compute classification report
    classification_rep = classification_report(y_true, y_pred, output_dict=True)
    # Display classification report in a structured way
    st.write("\nClassification Report:")
    classification_df = pd.DataFrame(classification_rep).transpose()
    st.dataframe(classification_df)

def main():
    st.title("ECG Image Classification Evaluation ")
    with st.spinner("Evaluating the model..."):
        # Evaluate the model
        evaluate_model()
    
if __name__ == "__main__":
    main()
