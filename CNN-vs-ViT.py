import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
import seaborn as sns
from keras.utils import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from keras.applications.resnet import ResNet50, preprocess_input as resnet_preprocess
from keras.applications.densenet import DenseNet169, preprocess_input as densenet_preprocess
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
#from tensorflow.keras.applications.vit import preprocess_input as vit_preprocess
from keras.models import Model
from vit_keras import vit, utils
import matplotlib.pyplot as plt

# set image directory
image_dir = './data'

# set model names
model_names = ['VGG-16', 'ResNet-50', 'DenseNet-169', 'ViT-B16', 'ViT-B32']
# define input shape of models
input_shape = (224, 224, 3)
# define batch size for image loading
batch_size = 32
# define number of classes
num_classes = len(os.listdir(image_dir))

# initialize empty dictionaries to hold features and labels
features = {}
labels = {}
# iterate over model names
for model_name in model_names:
    # load pre-trained model
    if model_name == 'VGG-16':
        model = VGG16(weights='imagenet', include_top=True)
        preprocess_fn = vgg_preprocess
    elif model_name == 'ResNet-50':
        model = ResNet50(weights='imagenet', include_top=True)
        preprocess_fn = resnet_preprocess
    elif model_name == 'DenseNet-169':
        model = DenseNet169(weights='imagenet', include_top=True)
        preprocess_fn = densenet_preprocess
    elif model_name == 'ViT-b16':
        model = vit.vit_b16(image_size=224, include_top=True, pretrained=True)
        #model = tf.keras.applications.ViT_B16(weights='imagenet', include_top=False)
        #preprocess_fn = utils.preprocess
    elif model_name == 'ViT-B32':
        model = vit.vit_b32(image_size=224, include_top=True, pretrained=True)
        #model = tf.keras.applications.ViT_L32(weights='imagenet', include_top=False)
        #preprocess_fn = utils.preprocess
        
    # create feature extractor
    #feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)

    feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)

    # initialize empty arrays to hold features and labels
    feature_list = []
    label_list = []

    ctr=0
    # iterate over image directory
    for class_folder in os.listdir(image_dir):
        class_path = os.path.join(image_dir, class_folder)

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # load and preprocess image
            image = load_img(image_path, target_size=input_shape[:2])
            image = img_to_array(image)
            image = preprocess_fn(image)
            image = np.expand_dims(image, axis=0)

            # extract features and add to list
            feature = feature_extractor.predict(image)
            feature = np.squeeze(feature)
            feature_list.append(feature)

            # add label to list
            label_list.append(class_folder)
            ctr += 1
            if ctr%1==0:
                print(f"[INFO]: Extracting features from {class_folder}")

    # convert feature and label lists to arrays
    features[model_name] = np.array(feature_list)
    labels[model_name] = np.array(label_list)

    print('Features extracted using', model_name)

    # encode labels as integers
    label_encoder = LabelEncoder()
    labels[model_name] = label_encoder.fit_transform(labels[model_name])

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features[model_name], labels[model_name], test_size=0.2, random_state=42)

    # train SVM on training data
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)

    # make predictions on testing data
    y_pred = svm.predict(X_test)

    # compute performance metrics
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print(report)

    # compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # plot confusion matrix using seaborn
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=num_classes, yticklabels=num_classes)
    # set plot labels and title
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'{model_name} Confusion Matrix')
    # save the plot to a file
    plt.savefig(f'./Results/{model_name}_confusion_matrix.png')

    # compute ROC curve and AUC
    y_pred_proba = svm.predict_proba(X_test)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # plot ROC curve
    plt.figure(figsize=(4, 4))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=label_encoder.classes_[i] + ' (AUC = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'./Results/{model_name}_ROC.png')
    #plt.show()

