
import os
import pandas as pd
import shutil

# Load the metadata
metadata_path = 'HAM10000_metadata.csv'
df = pd.read_csv(metadata_path)

# Create the output directory
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the unique classes
classes = df['dx'].unique()

# Create a directory for each class
for cls in classes:
    class_dir = os.path.join(output_dir, cls)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

# Move the images
images_dir = 'images'
for index, row in df.iterrows():
    image_id = row['image_id']
    image_class = row['dx']
    
    # Construct the full image path
    image_filename = f"{image_id}.jpg"
    src_path = os.path.join(images_dir, image_filename)
    dst_path = os.path.join(output_dir, image_class, image_filename)
    
    # Check if the source file exists before moving
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    else:
        print(f"Image not found: {src_path}")

print("Images have been organized into class folders.")
