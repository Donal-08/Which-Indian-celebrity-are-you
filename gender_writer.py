import os

# Set the path to the main folder containing the subfolders
folder_path = 'indian_celeb_archive/dataset/'

# Get a list of all subfolder names
subfolders = os.listdir(folder_path)

# Set the path for the output text file
output_file = 'indian_celeb_archive/actor_gender.txt'

# Open the text file in write mode
with open(output_file, 'w') as file:
    # Write each subfolder name to a new line in the text file
    for subfolder in subfolders:
        file.write(subfolder + '\n')

print('Subfolder names written to', output_file)
