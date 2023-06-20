import os
import pickle as pk

# Set the path to the main folder containing actor subfolders
data_path = 'indian_celeb_archive/dataset/'

celeb_and_filenames = []

# Iterate through images in the actor subfolders
for actor_folder in os.listdir(data_path):
    actor_folder_path = os.path.join(data_path, actor_folder)
    actor_name = actor_folder.replace("_", " ")                 # Replace underscore with space "Aamir_Khan" -> "Aamir Khan"
    for image_file in os.listdir(actor_folder_path):
        image_path = os.path.join(actor_folder_path, image_file)
        image_path = image_path.replace("\\", "/")       # Replace "\\" with "/" in the image path
        celeb_and_filenames.append((actor_name, image_path))      # Store both the actor name and image path


pk.dump(celeb_and_filenames, open('celeb_and_file_names.pkl', 'wb'))