import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

# ----- HELPER FUNCTIONS -------
def drop_null(dataset:dict) -> dict:
    """ Removes None data from the dataset """
    filtered_data, filtered_targets, filtered_paths = zip(*[(d,t,p) for d,t,p in zip(dataset['data'], dataset['target'], dataset['path'])
                                                            if d is not None])
    return {'data': np.array(filtered_data), 'target': filtered_targets, 'path': filtered_paths}

def plot_confusion_matrix(predicted_labels:dict, model:str, n_trials:int, with_colorbar:bool=False, save_fig:bool=False, title:str=None, ax=None):
    """
    Plot confusion matrix obtained by `model`. Summed across all `n_trials`

    -----------
    :param: predicted_labels: dictionary with classifiers (keys) and obtained results per class (values)
    :param: model: model to be assessed
    :param: n_trials: Total number of trials
    :param: save_fig: Boolean on whether to save plot or not
    :param: ax: can be used to subplot result
    """

    classes = np.unique(predicted_labels['true_labels'][0])

    # computing for sum of confusion matrices across trials
    cf = confusion_matrix(predicted_labels['true_labels'][0], predicted_labels[model][0])
    for idx in range(1, n_trials):
        cf = cf + confusion_matrix(predicted_labels['true_labels'][idx], predicted_labels[model][idx])

    # color palette
    cmp = mpl.colormaps['viridis']
    cmp = ListedColormap(cmp(np.linspace(0.25, 0.75, 128)))
    
    # plot confusion matrix and colorbar
    ax = ax or plt.gca()
    im = ax.imshow(cf, cmap=cmp)
    if with_colorbar:
        plt.colorbar(im, ax=ax)

    # change xticks to classes in the dataset
    ax.set_xticks(np.arange(len(classes)), labels=classes)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # x and y label, title
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title, fontsize=13)

    # create text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text_color = 'w'
            if i == j:
                text_color = 'k'
            if cf[i,j] != 0:
                ax.text(j, i, cf[i,j], ha="center", va="center", color=text_color, fontsize=6)

    if save_fig:
        plt.savefig(f'figures/{title}_result.png')

# ----- GLOBAL VARIABLES -------
save_classifiers = False
save_figs = True
estimators = [SVC(), RandomForestClassifier(), KNeighborsClassifier(), LinearDiscriminantAnalysis(), DecisionTreeClassifier()]
models = ['svm', 'rf', 'knn', 'lda', 'cart']
metrics = ['precision', 'recall', 'f1-score', 'specificity', 'support', 'accuracy']
predicted_labels = {mod: [] for mod in models+['true_labels']}
# ------------------------------

# load dataset: landmarks, targets, paths
dataset = np.load('./resources/alphabet_landmarks.npy', allow_pickle=True).item()
dataset = drop_null(dataset)

# initiate training
X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.10, 
                                                    stratify=dataset['target'], random_state=12)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# save correct labels
predicted_labels['true_labels'].append(np.array(y_test))

# test and predict
trained_models = []
for idx in range(len(estimators)):
    classifier = models[idx]
    print(f'{classifier} ---------------------')

    clsf = estimators[idx]
    clsf.fit(X_train, y_train)
    y_pred = clsf.predict(X_test)
    print("  Training and testing complete. ")
    print("  Classification report:")
    print(classification_report(predicted_labels['true_labels'][0],
                                y_pred))

    predicted_labels[classifier].append(y_pred)
    trained_models.append(clsf)

print()

# save trained models
if save_classifiers:
    for lab, mod in zip(models, trained_models):
        pickle.dump(mod, open(f"./trained_models/{lab}.sav", 'wb'))

if save_figs:
    for mod in models:
        plot_confusion_matrix(predicted_labels, model=mod, n_trials=1, with_colorbar=False, title=mod, save_fig=save_figs)