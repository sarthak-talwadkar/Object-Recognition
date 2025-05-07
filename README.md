# Object Recognition 

Real-Time Object Detection with OpenCV and Machine Learning

A real-time object detection system that identifies and classifies objects using custom image processing algorithms and machine learning techniques. Built with OpenCV, this project features interactive training, feature extraction, and classification using k-Nearest Neighbors (k-NN) and Nearest Neighbor algorithms.
Features 

    Real-Time Detection: Captures and processes live video feed from a camera.
    Custom Thresholding: Implements Otsu’s method and K-means clustering for adaptive image binarization.
    Noise Reduction: Uses morphological operations (erosion, dilation) to clean binary images.
    Feature Extraction: Computes shape descriptors like aspect ratio, eccentricity, orientation, and centroid.
    Machine Learning: Classifies objects using k-NN and Nearest Neighbor algorithms with a customizable training dataset.
    Interactive Training Mode: Add new object labels on-the-fly and update the training database (training_data.csv).
    Visualization: Draws bounding boxes, orientation arrows, and labels on detected regions.

Technologies Used 

    OpenCV: For image processing, video capture, and morphological operations.
    C++: Core implementation for performance-critical tasks.
    CSV Integration: Stores and loads training data for persistent learning.
    Custom Algorithms: Includes handcrafted implementations of Otsu’s thresholding, K-means clustering, and k-NN classification.

Prerequisites

    OpenCV 4.x
    C++17 compiler
    Webcam or video source

Training the Model 

    During execution, press N to capture the largest detected region.
    Enter a label for the object (e.g., "cup", "book").
    Features (percentage, aspect ratio, eccentricity) are saved to training_data.csv.
    The model updates in real-time for improved classification accuracy.

Contributing 

Contributions are welcome! Open an issue or submit a PR for:

    Optimizing performance.
    Enhancing classification accuracy.
    Adding support for new features or datasets.
