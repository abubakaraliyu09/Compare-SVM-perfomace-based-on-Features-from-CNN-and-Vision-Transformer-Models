Image Classification README

This is a Python script for training an SVM classifier using features extracted from pre-trained deep learning models. The script supports the following models:

    VGG-16
    ResNet-50
    DenseNet-169
    Vision Transformer B16 (ViT-B16)
    Vision Transformer B32 (ViT-B32)

Dependencies

The script requires the following packages:

    os
    numpy
    PIL
    sklearn
    tensorflow
    seaborn
    ke

    ras
    vit_keras
    matplotlib

How to Run

Before running the script, ensure you have all the necessary packages installed in your Python environment.

The script expects an image directory './data' where the subdirectories are class labels. Each subdirectory should contain the images for that class. For example, a dataset with two classes 'cats' and 'dogs' should have the following structure:

data/

├── cats/

│   ├── cat001.jpg

│   ├── cat002.jpg

│   └── ...

└── dogs/

    ├── dog001.jpg
    
    ├── dog002.jpg
    
    └── ...

To run the script:
python3 image_classification.py

The script will extract features from the images using the pre-trained models, train an SVM classifier on the extracted features, and evaluate the performance of the classifier using a variety of metrics. The results, including confusion matrices and ROC curves, will be saved to the './Results' directory.
Output

The script will generate the following output for each model:

    A classification report displaying precision, recall, f1-score, and support for each class
    A confusion matrix visualized using seaborn
    An ROC curve for each class

The confusion matrices and ROC curves will be saved to the './Results' directory in PNG format.
