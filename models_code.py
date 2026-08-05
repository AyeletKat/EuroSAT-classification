"""
This is the Final Project for the Machine Learning course at Ariel University, 2025.
It includes various machine learning models applied to the EuroSAT RGB version dataset.
The models include:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- AdaBoost
- Convolutional Neural Network (CNN) (implemented in a separate folder)

At the end you can also see fine tuning of resnet18

Enjoy!
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image  # Use PIL instead of tensorflow

# our own image loading functions
def load_img(img_path, target_size=None):
    """Load an image and optionally resize it to target_size"""
    img = Image.open(img_path)
    if target_size:
        img = img.resize(target_size)
    return img

def img_to_array(img):
    """Convert PIL Image to numpy array"""
    return np.array(img)



# if it's not the first run, load np arrays versions of data, save time
if os.path.exists('X_eurosat.npy') and os.path.exists('y_eurosat.npy'):
    print("a\n")
    X = np.load('X_eurosat.npy')
    y = np.load('y_eurosat.npy')
    print("b\n")

else:
    # Path to EuroSAT_RGB dataset
    data_dir = r'C:\Users\ayele\Documents\ML_2025\ML_project\EuroSAT_RGB' # change to your path

    images = []
    labels = [] # will be a list the size of all dataset - for i - images[i] is of category labels[i]

    # load images and labels
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                img = load_img(img_path, target_size=(64, 64))  # EuroSAT images are 64x64
                img_array = img_to_array(img)
                images.append(img_array.flatten())
                labels.append(label)
    # plot data distribution
    # import matplotlib.pyplot as plt
    # from collections import Counter
    # Plot data distribution with counts on top of bars and a blue-green colormap
    # label_counts = Counter(labels)
    # plt.figure(figsize=(10, 5))
    # bars = plt.bar(label_counts.keys(), label_counts.values(), color=plt.cm.viridis(np.linspace(0, 1, len(label_counts))))
    # plt.xlabel('Class Label')
    # plt.ylabel('Number of Samples')
    # plt.title('Class Distribution in EuroSAT Dataset')
    # plt.xticks(rotation=45)
    # # Add counts on top of bars
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.annotate(f'{int(height)}',
    #                  xy=(bar.get_x() + bar.get_width() / 2, height),
    #                  xytext=(0, 3),  # 3 points vertical offset
    #                  textcoords="offset points",
    #                  ha='center', va='bottom')
    # plt.tight_layout()
    # plt.show()

    # plot images representing each class
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    # for i, label in enumerate(np.unique(labels)):
    #     class_images = [img for img, lbl in zip(images, labels) if lbl == label]
    #     if class_images:
    #         axes[i // 5, i % 5].imshow(class_images[0].reshape(64, 64, 3).astype(np.uint8))
    #         axes[i // 5, i % 5].set_title(label)
    #         axes[i // 5, i % 5].axis('off')
    # plt.tight_layout()
    # plt.show()


    X = np.array(images)
    le = LabelEncoder() # it changes labels from string to sequential numbers, easier for models to work with
    y = le.fit_transform(labels)
    # print a dictionary of numbers and the labels they represent
    label_dict = {i: label for i, label in enumerate(le.classes_)}
    print("Label dictionary:", label_dict)

    # save processed arrays - images and labels as np arrays (to save loading time every run)
    np.save('X_eurosat.npy', X)
    np.save('y_eurosat.npy', y)

# splitting data into training and testing sets, here 80% for training and 20% for testing - - try other as changes tryed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def Hclustering(X_train, X_test, y_train, y_test):
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt
    # שלב 2: סטנדרטיזציה + PCA
    X_scaled = StandardScaler().fit_transform(X_train)
    X_pca = PCA(n_components=50).fit_transform(X_scaled)

    # שלב 3: דנדרוגרמה
    linked = linkage(X_pca, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=False,
            truncate_mode='level',
            p=30)
    plt.title("דנדרוגרמה - EuroSAT Hierarchical Clustering")
    plt.show()

    # שלב 4: מקבוץ היררכי בפועל
    n_clusters = 10
    model = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = model.fit_predict(X_pca)

    # שלב 5: הצגת תוצאה בצבעים
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', s=40)
    plt.title("מקבוץ היררכי על EuroSAT (PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()


def logisticRegression(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    iter = 200  # max_iter for convergence
    solverr = 'saga'  # using 'lbfgs' solver vs 'saga' vs `newton-cg`
    from sklearn.decomposition import PCA  # added pca to speed up training
    pca = PCA(n_components=100)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    model = LogisticRegression(solver=solverr, max_iter = iter)
    print(f"LR regular Model {solverr} solver, {iter} iterations started")
    model.fit(X_train, y_train)

    # evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"LR Model {solverr} solver, {iter} iterations train accuracy: {train_accuracy * 100:.2f}%")
    print(f"LR Model {solverr} solver, {iter} iterations test accuracy: {test_accuracy * 100:.2f}%")
    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Logistic Regression Confusion Matrix')
    plt.show()
    


def randomForest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from datetime import datetime
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)  # reduce dimensionality to speed up training
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    n_trees = 400  # num of trees in the forest
    print("RM pca {n_trees} TREES Model started")
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42 )
    model.fit(X_train, y_train)

    # evaluate
    accuracy = model.score(X_test, y_test)
    with open('svm_output.txt', 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - RM 200 TREES Model accuracy: {accuracy * 100:.2f}%\n")
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'RF Confusion Matrix')
    plt.show()


def SVM(X_train, X_test, y_train, y_test):
    from sklearn.svm import LinearSVC  # Using LinearSVC for linear kernel
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from datetime import datetime
    iter=200
    svm = LinearSVC(random_state=42, max_iter=iter ) # Using linear kernel - try other as changes tryed
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
    comps =100
    print("SVM Model, {iter} iters, {comps} comps started")

    pca = PCA(n_components=comps)  # Reduce dimensionality to speed up training
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    model = make_pipeline(scaler, pca, svm)  # Create a pipeline with PCA and SVM
    model.fit(X_train, y_train)

    # evaluate
    accuracy = model.score(X_test, y_test)
    # print output to svm_output.txt file
    with open('svm_output.txt', 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - SVM Model, {comps} pca, {iter} iters, accuracy: {accuracy * 100:.2f}%\n")
    print(f"SVM {comps} PCA, {iter} iters, Model accuracy: {accuracy * 100:.2f}%")

    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'SVM + PCA Confusion Matrix')
    plt.show()

def KNN(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.decomposition import PCA # add pca
    pca = PCA(n_components=15)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train, y_train)

    # evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'KNN Confusion Matrix')
    plt.show()
    

def adaboost(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from datetime import datetime
    from collections import Counter
    depth = 2  # max depth of the decision tree
    n_trees=100
    print("AdaBoost Model, {n_trees} trees, {depth} depth started")
    # add pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)  # Reduce dimensionality to speed up training
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    # using Decision Tree Classifier as the base estimator
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators=n_trees,  # Number of trees in the ensemble
        random_state=42)
    model.fit(X_train, y_train)
    # print loss and accuracy for train and test sets
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"AdaBoost Model train accuracy: {train_accuracy * 100:.2f}%")
    print(f"AdaBoost Model test accuracy: {test_accuracy * 100:.2f}%")    
    with open('adaboost_output.txt', 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AdaBoost Model, {n_trees} trees, {depth} depth, accuracy: {test_accuracy * 100:.2f}%\n")

    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'AdaBoost Confusion Matrix')
    plt.show()





# RUNNING MODELS!

# Hclustering(X_train, X_test, y_train, y_test)
# logisticRegression(X_train, X_test, y_train, y_test) # first run - Model accuracy: 38.85%
# randomForest(X_train, X_test, y_train, y_test) # first run 100 trees - Model accuracy: 69.04%
# SVM(X_train, X_test, y_train, y_test) # ran over 30 minutes, didn't finish
# KNN(X_train, X_test, y_train, y_test) # first run - Model accuracy: 34.44% \ with pca 49+
# adaboost(X_train, X_test, y_train, y_test) # first run 10 trees- Model accuracy: 25.50


# running resnet on the dataset - finetuning a pre-trained reset18
# import torch
# import torchvision
# from torchvision import datasets, transforms
# from torchvision.models import resnet18
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score

# def train_resnet():
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     data_dir = r'C:\Users\ayele\Documents\ML_2025\ML_project\EuroSAT_RGB'
#     dataset = datasets.ImageFolder(root=data_dir, transform=transform)
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     model = resnet18(pretrained=True)
#     num_classes = len(dataset.classes)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     num_epochs = 10
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         for images, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         train_loss /= len(train_loader)

#         # Validation loss calculation
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#         val_loss /= len(val_loader)

#         print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#     # Final validation accuracy
#     model.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for images, labels in val_loader:
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#     accuracy = accuracy_score(all_labels, all_preds)
#     print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")

#     return model

# model = train_resnet()

# # Save and reload
# torch.save(model.state_dict(), 'resnet_eurosat.pth')
# model = resnet18()
# model.fc = nn.Linear(model.fc.in_features, 10)
# model.load_state_dict(torch.load('resnet_eurosat.pth'))
# model.eval()
# print("ResNet model loaded and ready for inference.")






# import matplotlib.pyplot as plt

# train_loss = [0.5011, 0.2831, 0.2179, 0.1760, 0.1487, 0.1267, 0.1285, 0.0913, 0.0957, 0.0733]
# val_loss   = [0.4553, 0.3075, 0.2201, 0.2328, 0.1664, 0.2174, 0.2151, 0.2138, 0.1850, 0.2017]

# epochs = list(range(1, 11))

# plt.figure(figsize=(8, 5))
# plt.plot(epochs, train_loss, label='Train Loss', marker='h')
# plt.plot(epochs, val_loss, label='Validation Loss', marker='H', color='orange')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('ResNet18 FineTuning Training and Validation Loss')
# plt.legend()
# plt.grid(True)
# plt.xticks(epochs)
# # Annotate each epoch with the higher loss above and the lower loss below
# for i, epoch in enumerate(epochs):
#     t_loss = train_loss[i]
#     v_loss = val_loss[i]
#     if i == len(epochs) - 1:
#         # Last point: print train_loss closer (-12)
#         if t_loss > v_loss:
#             plt.annotate(f'{t_loss:.4f}', (epoch, t_loss), xytext=(0, 8), textcoords='offset points', ha='center')
#             plt.annotate(f'{v_loss:.4f}', (epoch, v_loss), xytext=(0, -18), textcoords='offset points', ha='center', color='black')
#         else:
#             plt.annotate(f'{v_loss:.4f}', (epoch, v_loss), xytext=(0, 8), textcoords='offset points', ha='center', color='black')
#             plt.annotate(f'{t_loss:.4f}', (epoch, t_loss), xytext=(0, -12), textcoords='offset points', ha='center')
#     else:
#         if t_loss > v_loss:
#             plt.annotate(f'{t_loss:.4f}', (epoch, t_loss), xytext=(0, 8), textcoords='offset points', ha='center')
#             plt.annotate(f'{v_loss:.4f}', (epoch, v_loss), xytext=(0, -18), textcoords='offset points', ha='center', color='black')
#         else:
#             plt.annotate(f'{v_loss:.4f}', (epoch, v_loss), xytext=(0, 8), textcoords='offset points', ha='center', color='black')
#             plt.annotate(f'{t_loss:.4f}', (epoch, t_loss), xytext=(0, -18), textcoords='offset points', ha='center')
# plt.tight_layout()
# plt.show()
