import os
import json
import paddle
from paddleocr import PaddleOCR
import cv2
from paddle.io import Dataset
from paddle.vision.transforms import ToTensor, Compose
from paddle.optimizer import Adam
import paddle.nn.functional as F
import numpy as np

# Define Expiry Date Dataset class
class ExpiryDateDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.annotations = json.load(open(annotation_file, 'r'))
        self.img_dir = img_dir
        self.transform = transform
        self.data = list(self.annotations.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, annotations = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        ann = annotations[idx]

        # Preprocess image (apply transforms if needed)
        if self.transform:
            image = self.transform(image)

        # Extract bboxes and transcriptions
        bboxes = [obj['bbox'] for obj in ann if 'transcription' in obj]
        transcriptions = [obj['transcription'] for obj in ann if 'transcription' in obj]

        return image, bboxes, transcriptions

# Load PaddleOCR Pre-trained Model
def load_model():
    ocr_model = PaddleOCR(use_angle_cls=True, lang="en")  # Automatically downloads the pre-trained model
    return ocr_model

# Custom function to extract expiry dates and compare with ground truth
def evaluate_model(ocr_model, data_loader):
    total_correct = 0
    total_samples = 0

    for images, bboxes, transcriptions in data_loader:
        # Convert images to numpy for OCR input
        images_np = [image.numpy().transpose(1, 2, 0) for image in images]  # Convert to HWC format

        # Forward pass: OCR model extracts text
        for i, image in enumerate(images_np):
            result = ocr_model.ocr(image, cls=True)

            # Extract predicted texts
            predicted_texts = [word_info[1][0] for line in result for word_info in line]

            # Compare with target transcriptions
            if len(predicted_texts) == len(transcriptions[i]):
                total_correct += sum([1 for pred, target in zip(predicted_texts, transcriptions[i]) if pred == target])
            total_samples += len(transcriptions[i])

    accuracy = total_correct / total_samples * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Save the fine-tuned model
def save_model(ocr_model, save_path):
    paddle.jit.save(ocr_model, save_path)
    print(f"Model saved at {save_path}")

# Main Function to train and save the model
if __name__ == "__main__":
    # Define paths
    annotation_file = 'annotations.json'  # Path to your annotations file
    img_dir = "C:\\Users\\T.Reddy\\Downloads\\Products-Real\\Products-Real\\train\\images"  # Path to your images directory
    model_save_path = './expiry_date_model'  # Path where you want to save the model

    # Load dataset
    dataset = ExpiryDateDataset(
        annotation_file=annotation_file,
        img_dir=img_dir,
        transform=Compose([ToTensor()])  # Convert image to tensor
    )

    data_loader = paddle.io.DataLoader(dataset, batch_size=8, shuffle=False)

    # Load the pre-trained PaddleOCR model
    ocr_model = load_model()

    # Evaluate the model
    print("Evaluating model on expiry date extraction...")
    evaluate_model(ocr_model, data_loader)

    # Save the model for future use
    save_model(ocr_model, model_save_path)
