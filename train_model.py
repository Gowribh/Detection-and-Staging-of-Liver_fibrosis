import tensorflow as tf
import os
import keras.saving
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure dataset path is correct
dataset_path = os.path.join(os.getcwd(), "dataset")
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

# Load dataset
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',  # ✅ Use 'categorical' instead of 'binary'
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',  # ✅ Ensure consistency
    subset='validation'
)

# Check class indices (optional)
print(f"Class indices: {train_generator.class_indices}")

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # ✅ 4 output classes for fibrosis stages
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# # Save model
# model.save('model/fibrosis_model.h5')
# model.save('model/fibrosis_model.keras')  # Saves in the new format
keras.saving.save_model(model, 'model/fibrosis_model.keras')
