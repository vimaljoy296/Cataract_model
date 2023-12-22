# import Necessary Libraries 
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Set the path to your dataset
dataset_dir = "C:/Cognitive Learning/Dataset"

# Get a list of all image files in the dataset directory
all_images = []
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".png"):
            all_images.append(os.path.join(root, file))

print(f"Number of images: {len(all_images)}")

# Create labels based on folder names (assuming binary classification)
labels = [1 if "cataract" in img else 0 for img in all_images]

# Create a DataFrame with file paths and labels
df = pd.DataFrame({'filename': all_images, 'label': labels})

# Convert labels to strings
df['label'] = df['label'].astype(str)

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Count the number of images in each category
normal_count = df[df['label'] == '0'].shape[0]
cataract_count = df[df['label'] == '1'].shape[0]

print(f"Number of Normal Images: {normal_count}")
print(f"Number of Cataract Images: {cataract_count}")

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

 #Load and augment training data
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Load and rescale test data
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Increase model complexity with additional layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Increase the number of epochs
epochs = 20

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df) // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(test_df) // batch_size
)

# Evaluate the model
accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy[1]*100:.2f}%")

# Save the trained model with a different name
model.save('New_cataract_model.h5')

# Alternatively, you can save it in the same folder as the dataset
model.save(os.path.join(dataset_dir, 'New_cataract_model.h5'))