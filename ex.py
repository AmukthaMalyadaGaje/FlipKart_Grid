import os
import json

# Specify the directory containing the image files
directory_path = 'C:\\Users\\devad\\OneDrive\\Desktop\\Flipkart Grid\\shelf_life_prediction\\dataset\\brand_images\\Logos'

# Initialize an empty dictionary to hold brand and image file mappings
brand_data = {}

# Supported image file extensions
image_extensions = ('.jpg', '.jpeg', '.png')

# Read each file in the specified directory
for filename in os.listdir(directory_path):
    # Check if the file is an image
    if filename.lower().endswith(image_extensions):
        # Get the brand name without extension
        brand_name = os.path.splitext(filename)[0]

        # Initialize a list for the brand if it doesn't exist
        if brand_name not in brand_data:
            brand_data[brand_name] = []

        # Append the image filename to the list for this brand
        brand_data[brand_name].append(filename)

# Specify the output JSON file path
output_file_path = 'brand_image_mapping.json'

# Write the dictionary to a JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(brand_data, json_file, indent=4)

print(
    f"File '{output_file_path}' created successfully with brand and image mappings.")
