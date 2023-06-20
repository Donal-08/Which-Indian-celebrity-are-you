#!pip install streamlit
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle as pk
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
import heapq
from facenet_pytorch import MTCNN
import numpy as np



face_detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
celeb_feature_list = np.array(pk.load(open('embedding.pkl', 'rb')))
celeb_and_file_names = pk.load(open('celeb_and_file_names.pkl', 'rb'))


def save_uploaded_image(uploaded_image):
    """
    Saves an uploaded image to the 'uploads' directory.
    
    Parameters:
        uploaded_image (FileStorage): The uploaded image file.
        
    Returns:
        bool: True if the image is successfully saved, False otherwise.
                                                                            """
    try:
        uploads_dir = 'uploads'
        
        # Create the 'uploads' directory if it doesn't exist
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        with open(os.path.join(uploads_dir, uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        
        return True
    except:
        return False


def extract_features(person_image_path, model, face_detector):

    """    
        Extracts facial features from an input image using a pre-trained model.
        It first detects the face from  the image using a face_detector model,
        then extracts its features using a model

        Params:
            person_image_path (str): The file path to the input image.
            model (tf.keras.Model): The pre-trained model for feature extraction.
            face_detector: The face detection model 

        Returns:
            numpy.ndarray: The extracted facial features as a 1D numpy array. 
            Node: if face can not be cropped using the model                                
                                                                                    """
    
    sample_img = cv2.imread(person_image_path)
    boxes, _ = face_detector.detect(sample_img)

    if boxes is not None and len(boxes) > 0:
        bounding_box = boxes[0].astype(int)  # Convert float values to integers
        x_low, y_low, x_high, y_high = bounding_box

        # Crop the face region from the image
        cropped_face = sample_img[y_low:y_high, x_low:x_high]

    else: 
        return None

    image = Image.fromarray(cropped_face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    face_features = model.predict(preprocessed_img).flatten()

    return face_features


# Function to find the celebrity a person looks like
def find_celebrity(celeb_feature_list ,face_features):
    """
        Finds the celebrity that a person looks like based on facial similarity scores.

    Parameters:
        celeb_feature_list (list): A list of celebrity feature vectors.
        face_features (numpy.ndarray): The feature vector of the person's face.

    Returns:
        tuple: A tuple containing the name of the matched celebrity and the image path of the matched celebrity.
    """

    similarities = []
    
    # Find the cosine similarity of current image with all of the 13000 image_features
    for celeb_feature in celeb_feature_list:
        similarity = cosine_similarity(face_features.reshape(1, -1), celeb_feature.reshape(1, -1))[0][0]
        similarities.append(similarity)
    
    # Get the indices of the top 5 similarity scores
    top_indices = heapq.nlargest(5, range(len(similarities)), similarities.__getitem__)

    # Dictionary to store the weighted similarity scores for each celebrity
    weighted_similarity_scores = {}
    # Dict to store the celebrity name and corresponding image path
    celeb_name2path = {}

    # Iterate over the top 5 indices
    for i in top_indices:
        # Get the similarity score and celebrity name
        similarity_score = similarities[i]
        celebrity_name = celeb_and_file_names[i][0]
        img_path = celeb_and_file_names[i][1]
        
        # Add the img path to the corresponding celebrity name
        if celebrity_name not in celeb_name2path:
            celeb_name2path[celebrity_name] = img_path

        # Add the similarity score to the corresponding celebrity's weighted similarity score
        if celebrity_name in weighted_similarity_scores:
            weighted_similarity_scores[celebrity_name] += similarity_score
        else:
            weighted_similarity_scores[celebrity_name] = similarity_score


    # Get the celebrity with the highest weighted similarity score
    matched_celebrity = max(weighted_similarity_scores, key=weighted_similarity_scores.get)
    matched_image_path = celeb_name2path[matched_celebrity]

    return matched_celebrity,  matched_image_path



# The APP layout
st.title('Which bollywood celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # save the image in a directory
    if save_uploaded_image(uploaded_image):
        # load the image
        display_image = Image.open(uploaded_image)

        # extract the features
        features = extract_features(os.path.join('uploads', uploaded_image.name),model, face_detector)
        if features is not None:
            # recommend
            matched_celebrity, matched_imagepath = find_celebrity(celeb_feature_list, features)
            # display
            col1, col2 = st.columns(2)

            with col1:
                st.header('Your uploaded image')
                st.image(display_image)
            with col2:
                st.header("Looks like " + matched_celebrity)
                st.image(matched_imagepath, width=250)
        
        else:
            st.write("No Face could be detected. \n Possible reasons are \n- Image is too small or of low quality for face detection. \n- Image does not contain recognizable faces")

    else:
        st.subheader("Error! File could not be saved ")

else:
    st.subheader("Please Upload an image file.")