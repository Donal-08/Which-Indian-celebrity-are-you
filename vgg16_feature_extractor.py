import numpy as np
import pickle as pk
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input


# Function to Extract features from an image
def extract_features(image_path, base_model):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    expanded_image = np.expand_dims(image, axis=0)
    preprocessed_image = preprocess_input(expanded_image)

    features = base_model.predict(preprocessed_image)
    features = np.array(features).flatten()
    return features

# Load the dumped celebrity and filenames
celeb_and_file_names = pk.load(open('celeb_and_file_names.pkl', 'rb'))

# DEFINING THE MODEL
# Load the pre-trained VGG16 model (excluding the top classification layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

print(base_model.summary())