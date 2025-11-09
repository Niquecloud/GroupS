# GroupS
Capstone Project FAW


Capstone Project: Fall Armyworm (FAW) Binary Classification

This project demonstrates a complete pipeline for detecting the presence of Fall Armyworm (FAW) in crop images using a Convolutional Neural Network (CNN) with PyTorch. The workflow covers dataset preparation, preprocessing, training, fine-tuning, and exporting the model for deployment.

1. Setup & Library Installation

All required libraries are installed to support deep learning, image processing, and model deployment:

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm albumentations==1.3.0 pandas scikit-learn matplotlib pillow onnx onnxruntime torchmetrics


Key libraries used:

PyTorch – model building and training

Albumentations – image augmentation

Pandas & scikit-learn – data handling and metrics

ONNX / ONNX Runtime – model export and deployment

2. Dataset Mounting & Verification

Google Drive is mounted and dataset paths verified:

base_path = "/content/drive/MyDrive/Capstone_Project_FAW_Dataset_v4.1"
pos_path = base_path + "/Positive_Faw_Dataset"
neg_path = base_path + "/Negative_Faw_Dataset"


The total number of images in both positive and negative classes is counted, ensuring dataset connectivity and integrity.

3. Standardize Filenames & Convert to .jpg

All images are standardized and converted to .jpg format for consistency:

standardize_and_convert(pos_path, "FAW_POS")
standardize_and_convert(neg_path, "FAW_NEG")


Each image is opened, converted to RGB, renamed with a prefix, and saved as .jpg.

Original files are removed after conversion.

4. CSV Metadata Generation

A CSV file is created to map each image to its class label and metadata:

generate_csv_metadata(base_path, csv_output_path)


Each image is labeled 1 for FAW positive and 0 for FAW negative.

Additional metadata includes crop type, growth stage, and scenario type (extracted from folder structure).

5. Train/Test Split

The dataset is split into training (75%) and testing (25%) sets, maintaining class distribution:

train_df, test_df = train_test_split(df, test_size=0.25, stratify=df['class_label'], random_state=42)


CSV files for training and testing are saved for reproducibility.

6. Image Preprocessing & Augmentation

All images are preprocessed and augmented to improve model generalization:

preprocess_images(train_df, preprocessed_train_folder, augment=True)
preprocess_images(test_df, preprocessed_test_folder, augment=True)


Preprocessing steps:

Resize images to 224x224 pixels.

Normalize pixel values (0-1).

Augment each image 10 times using random flips, rotations, brightness/contrast adjustments, and color modifications.

7. CNN Model Training

A CNN model is trained using a pre-trained ResNet18 modified for binary classification:

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)


Loss function: BCEWithLogitsLoss

Optimizer: Adam (lr=1e-4)

Batch size: 32

Epochs: 10

Training loop computes loss and accuracy for each epoch.

8. Initial Model Evaluation

The model is evaluated on the test set using standard metrics:

Test Accuracy: 0.8190
Test Precision: 0.9302
Test Recall: 0.6897
Test F1-score: 0.7921


Accuracy measures overall correctness.

Precision measures correctness of positive predictions.

Recall measures the ability to detect positive cases.

F1-score is the harmonic mean of precision and recall.

9. Fine-Tuning for Improved Recall

Fine-tuning is performed by augmenting the positive class more aggressively and adjusting the prediction threshold:

threshold = 0.4


Only positive-class images are augmented to improve detection of FAW.

BCEWithLogitsLoss is weighted to account for class imbalance.

Training runs for 5 additional epochs.

10. Fine-Tuned Model Evaluation

After fine-tuning, the final metrics improved significantly:

Fine-tuned Test Accuracy: 0.8879
Fine-tuned Test Precision: 0.9091
Fine-tuned Test Recall: 0.8621
Fine-tuned Test F1-score: 0.8850

Confusion Matrix:
[[53  5]
 [ 8 50]]


True Positives: 50

True Negatives: 53

False Positives: 5

False Negatives: 8

Fine-tuning balanced the trade-off between precision and recall, resulting in better detection of FAW-positive images.

11. Export Model to ONNX

The fine-tuned model is exported for deployment using ONNX:

torch.onnx.export(model, dummy_input, onnx_export_path, export_params=True, opset_version=12, ...)


Ensures compatibility with multiple platforms and inference environments.

12. Deploy & Inference with ONNX

The ONNX model can predict the FAW class for new images:

label, prob = predict_image_onnx(sample_image_path)
print(f"Predicted Label: {label} | Probability: {prob:.4f}")


Images are preprocessed to 224x224, normalized, and passed through the model.

Predictions use a threshold of 0.4 for positive detection.

✅ Final Outcome

Model is trained, fine-tuned, and deployed.

Achieves 88.79% accuracy with high recall and balanced precision.

Confusion matrix confirms reliable binary classification for FAW detection.
