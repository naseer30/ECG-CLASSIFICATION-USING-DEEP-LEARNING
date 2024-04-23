import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, RepeatVector
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Define directories for train and validation data 
train_data_dir = './PreprocessedData/train'
validation_data_dir = './PreprocessedData/validation'

# Configure data generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Set image dimensions
target_size = (150, 150)

# Define the generator for training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    shuffle=True,
    subset='training'  # Indicate that we are using only the training subset
)

# Define the generator for validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    shuffle=False
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    RepeatVector(50),  # Repeat the features to match the number of timesteps
    LSTM(64, return_sequences=True),  # LSTM layer with 64 units and return sequences
    Dropout(0.5),  # Dropout layer to prevent overfitting
    LSTM(64),  # Second LSTM layer with 64 units
    Dropout(0.4),
    Dense(512, activation='relu'),  # Dense layer with 512 units
    Dense(4, activation='softmax')  # Output layer with softmax activation for 4 classes
])

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the directory to save the models
checkpoint_dir = "Model_Checkpoints"

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Define the checkpoint filename
checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.h5")

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,  # Save only the model weights
    save_freq='epoch',  # Save at the end of each epoch
    verbose=1  # Print verbose messages
)
# Define a callback to save the best model based on validation accuracy
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)

# Define a custom callback to save training history for each epoch
class SaveHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        np.save(f'train_history_test_epoch_{epoch+1}.npy', self.model.history.history)

# Train the model with early stopping and checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=15,  # Reduce the number of epochs
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping, checkpoint_callback,[checkpoint], SaveHistory()],
    
)

# Evaluate the model
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print("Validation Loss:", validation_loss)
print("Validation Accuracy:", validation_accuracy)

# Save the final model
model.save('my_CRNN.h5')
print("Model saved successfully.")

