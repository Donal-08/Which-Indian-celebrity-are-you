import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.applications.vgg16 import VGG16



def load_image(image_path, target_size=(64, 64)):
    image = load_img(image_path, target_size=target_size)
    return image

def preprocess_image(image):
    image = img_to_array(image)
    image = image / 255.0  # Normalize pixel values between 0 and 1

    return image


def train_model(pre_trained_model, X_train, y_train, epochs=5, batch_size=32, validation_data=None):
    # Freeze the weights of the pre-trained layers so they are not updated during training
    for layer in pre_trained_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(pre_trained_model)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    return model

def evaluate_model(trained_model, X_test, y_test):
    loss, accuracy = trained_model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    y_pred_prob = trained_model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    predicted_labels = label_encoder.inverse_transform(y_pred.flatten())

    report = classification_report(label_encoder.inverse_transform(y_test), predicted_labels)
    print(report)


def predict_gender(trained_model, image):
    image = np.expand_dims(image, axis=0)
    prediction = trained_model.predict(image)
    y_pred = (prediction > 0.5).astype(int)
    predicted_label = label_encoder.inverse_transform(y_pred.flatten())

    return predicted_label[0]


# Set the path to the main folder containing actor subfolders
data_path = 'indian_celeb_archive/dataset/'

# Initialize lists to store images and corresponding labels
images = []
labels = []

# Read the actor gender information from the text file
gender_file = 'indian_celeb_archive/actor_gender.txt'

with open(gender_file, 'r') as file:
    for line in file:
        actor, gender = line.strip().split()

        # Iterate through images in the actor subfolder
        actor_folder_path = os.path.join(data_path, actor)       #  eg. 'indian_celeb_archive/dataset/{ACTOR_NAME}'
        for image_file in os.listdir(actor_folder_path):          # eg. 'ACTOR_NAME.1.jpg'
            image_path = os.path.join(actor_folder_path, image_file)   # eg. 'indian_celeb_archive/dataset/{ACTOR_NAME}/{ACTOR_MAME.1.jpg}'

            # Convert backslashes to forward slashes in the image path
            image_path = image_path.replace("\\", "/")

            # Load the image
            image = load_image(image_path)
            image = preprocess_image(image)

            # Add the image and corresponding label to the lists
            images.append(image)
            labels.append(gender)

# Preprocess the images
images = np.array(images)
labels = np.array(labels)

# Perform label encoding on the gender labels i.e convert labels to 0-1
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load the pre-trained VGG16 model (excluding the top classification layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Train the model
trained_model = train_model(base_model, X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
evaluate_model(trained_model, X_test, y_test)


# Make predictions on a sample images
sample_image_path = 'test_data/pushpaposter_female_look.jpeg'
sample_image = load_image(sample_image_path)
predicted_gender = predict_gender(trained_model, sample_image)
print(f"Predicted gender: {predicted_gender}")


# Define the path to the folder containing the sample images
sample_images_folder = 'test_data/'

# Loop through each image file in the sample images folder
for image_file in os.listdir(sample_images_folder):
    # Construct the full path to the image
    image_path = os.path.join(sample_images_folder, image_file)

    # Load the image
    sample_image = load_image(image_path)

    # Predict the gender of the sample image
    predicted_gender = predict_gender(trained_model, sample_image)

    # Extract the actor name from the image file name
    actor_name = image_file.split('.')[0]

    # Print the prediction result
    print(f"The model thinks {actor_name} is a {predicted_gender}")