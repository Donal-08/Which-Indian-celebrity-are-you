import numpy as np
import pickle as pk
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from tqdm import tqdm

# Function to Extract features from an image
def extract_features(image_path, base_model):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    expanded_image = np.expand_dims(image, axis=0)
    preprocessed_image = preprocess_input(expanded_image)

    features = base_model.predict(preprocessed_image)
    features = np.array(features).flatten()
    return features


# Load the celeb and file names
celeb_and_file_names = pk.load(open('celeb_and_file_names.pkl', 'rb'))

# loading the pre-trained VGGFace model (excluding the top classification layer)
vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling="avg")


# print(vgg_model.summary())

# features = []

# for celeb, filename in tqdm(celeb_and_file_names):
#     features.append(extract_features(filename, vgg_model))

# pk.dump(features, open('embedding.pkl', 'wb'))