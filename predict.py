# Load the image - First FACE DETECTION 
# Extract its features 
# Find the minimum cosine distance of the current image with all the 12257 features
# Output that image

import cv2
import numpy as np
import torch
import os
from facenet_pytorch import MTCNN
import pickle as pk
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import heapq


celeb_feature_list = np.array(pk.load(open('embedding.pkl', 'rb')))
celeb_and_file_names = pk.load(open('celeb_and_file_names.pkl', 'rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling="avg")

face_detector = MTCNN()

folder_path = 'test_data/'

for filename in os.listdir(folder_path):
    # the path to the person's image
    person_image_path = os.path.join(folder_path, filename)
    print(person_image_path)

    sample_img = cv2.imread(person_image_path)

    boxes, _ = face_detector.detect(sample_img)

    if boxes is not None and len(boxes) > 0:
        bounding_box = boxes[0].astype(int)  # Convert float values to integers
        x_low, y_low, x_high, y_high = bounding_box

        # Crop the face region from the image
        cropped_face = sample_img[y_low:y_high, x_low:x_high]

    else: 
        print("No Face could be detected. \n Possible reasons are \n- image is too small or of low quality for face detection. \n  image does not contain recognizable faces")


    image = Image.fromarray(cropped_face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    face_features = model.predict(preprocessed_img).flatten()
    # print(face_features.shape)

    # Function to find the celebrity a person looks like
    def find_celebrity(face_features):
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


    # Find the celebrity the person looks like
    matched_celebrity, matched_image_path = find_celebrity(face_features)
    print(matched_celebrity)
    print(matched_image_path)

    # temp_img = cv2.imread(matched_image_path)
    # cv2.imshow('output', temp_img)
    # cv2.waitKey(0)