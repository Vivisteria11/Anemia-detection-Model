# Anemia-detection-Model
📌 AI-Based Anemia Detection Using MobileNetV2
This project presents an AI-powered anemia detection system that classifies eye conjunctiva images into Anemic or Normal categories using transfer learning with MobileNetV2. The solution is designed to be lightweight, accurate, and deployable on edge devices via TensorFlow Lite (TFLite).
Accuracy of 92.5% achieved.

✅ Key Features
🔍 Model: Pretrained MobileNetV2 backbone with custom classification layers.

🎯 Task: Binary classification – Anemic vs. Non-Anemic.

⚙️ Image Preprocessing:

Augmented training data using rotation, zoom, shift, flip, and shear.

Normalization with rescale (1./255).

🧠 Training Configuration:

Frozen first 120 layers of MobileNetV2 for faster convergence.

Optimizer: Adam with learning rate 1e-4.

Early stopping, learning rate reduction, and model checkpointing.

📈 Evaluation:

Accuracy and AUC on train, validation, and test sets.

Visualization of training history.

📦 Export:

Final model saved as .h5.

Optimized model converted to .tflite for deployment.

In progress to be integrated with a web application.
