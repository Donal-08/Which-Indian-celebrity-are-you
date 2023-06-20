# Which Indian Celebrity are you?

## Motivation
The motivation behind building this project was to create a fun and interactive application that allows users to find out which Indian celebrity they resemble the most based on their facial features. Indian celebrities are widely recognized and admired, and I saw this fact being utilised in Facebook, where you have such kind of things going on to engage people. This project aims to provide an entertaining and engaging experience for users by leveraging facial recognition and similarity matching techniques. The dataset I used for training is available in [Kaggle's Bollywood celeb localized face dataset (extended)](https://www.kaggle.com/datasets/sroy93/bollywood-celeb-localized-face-dataset-extended)

## Problem Statement
The project addresses the question of "Which Indian Celebrity are you?" by analyzing the facial features of a person and comparing them with the facial features of various Indian celebrities. By using advanced facial recognition algorithms and similarity metrics, the project determines the closest match and identifies the celebrity that the person resembles the most.

## Unique Features
The project offers several standout features that set it apart:
- Face Detection: The application uses advanced face detection algorithms to accurately detect and locate faces in the uploaded image, before going into feature extraction
- Feature Extraction: Facial features are extracted using state-of-the-art deep learning models(here VGGFace), capturing unique characteristics of individuals.
- Similarity Matching: The project utilizes cosine similarity to measure the resemblance between a person's facial features and those of Indian celebrities.
- Weighted Similarity: Instead of considering only the highest similarity score, the project calculates a weighted similarity score to identify the closest match among the top 5 candidates. The weighted similarity score is obtained by summing up the similarity scores of a celebrity from all the 5 matches. 
- Image Visualization: The application provides visual feedback by displaying the matched celebrity's name and an image for easy comparison.

## Installation and Setup
1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Download the pre-trained face detection and face feature extractor models.
4. Launch the application by running `streamlit run app.py`.
5. Now you can use the application and upload an image to test

## How to use
1. Upload an image of a person to the application.
2. The application will process the image and identify the closest matching Indian celebrity.
3. The name of the celebrity and an image for comparison will be displayed.
4. Users can enjoy discovering which Indian celebrity they resemble the most.

## Possible Future Enhancements
The project has tremendous potential for expansion and improvement. Some features that could be implemented in the future include:
- Better Models: In future, I plan to experiment with different combinations of face detection algorithms and image feature extractors.
    - Face Detection Packages: [Dlib](https://pypi.org/project/dlib/), [MTCNN(Multi-task Cascaded Convolutional Networks)](https://pypi.org/project/mtcnn/), [FaceNet-pytorch]( https://pypi.org/project/facenet-pytorch/)
    - Image Feature Extraction Models: VGGFace, FaceNet, DeepFace
- More Celebrities: Include more celebrities in the training set   
- Dynamic Database: Updating the celebrity database regularly to include new celebrities and improve matching accuracy.
- Multiple Celebrity Matches: Providing the top few celebrity matches with their respective similarity scores.
- User Feedback: Allowing users to rate and provide feedback on the accuracy of the matches.

