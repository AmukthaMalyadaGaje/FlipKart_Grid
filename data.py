import json

# Load the annotations from a file
def load_annotations(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Convert the loaded annotations to the required format
def convert_annotations(raw_annotations):
    output_data = {}
    for img_name, annotations in raw_annotations.items():
        # Only keep annotations that have a transcription
        for ann in annotations["ann"]:
            if "transcription" in ann:
                output_data[img_name]= ann["transcription"]
    
    return output_data

# Save the converted annotations to a new JSON file
def save_annotations(output_data, output_file_path):
    with open(output_file_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)
    print(f"Annotations have been converted and saved to {output_file_path}.")

raw_annotations_file_path = 'C:\\Users\\T.Reddy\\Downloads\\Products-Real\\Products-Real\\train\\annotations.json'  # Path to your raw annotations file
output_annotations_file_path = 'annotations_converted.json'  # Path to save the converted annotations

raw_annotations = load_annotations(raw_annotations_file_path)
converted_annotations = convert_annotations(raw_annotations)
save_annotations(converted_annotations, output_annotations_file_path)
