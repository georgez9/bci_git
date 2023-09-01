import pandas as pd  # pandas is used to load and manipulate data and for One-Hot Encoding
import numpy as np  # numpy is used to calculate the mean and standard deviation
import matplotlib.pyplot as plt  # matplotlib is for drawing graphs
import matplotlib.colors as colors
from sklearn.model_selection import train_test_split  # split  data into training and testing sets
from sklearn import preprocessing  # scale and center data
from sklearn.svm import SVC  # this will make a support vector machine for classificaiton
from sklearn.model_selection import GridSearchCV  # this will do cross validation
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score  # creates and draws a confusion matrix
from sklearn.decomposition import PCA  # to perform PCA to plot the data
from joblib import dump


def svm_train():
    column_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    df0 = pd.read_csv("./data/state1_cvt.txt", header=None, names=column_names)
    df0["state"] = 1
    df1 = pd.read_csv("./data/state2_cvt.txt", header=None, names=column_names)
    df1["state"] = 0
    df = pd.concat([df0, df1], ignore_index=True)
    X_encoded = df.drop('state', axis=1).copy()
    y = df['state'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)
    scaler = preprocessing.StandardScaler().fit(X_train)
    dump(scaler, './model/scaler.joblib')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_features = np.size(X_train_scaled, axis=1)
    param_grid = [
        {'C': [1, 10, 100, 1000],
         'gamma': [1 / num_features, 1, 0.1, 0.01, 0.001, 0.0001],
         'kernel': ['rbf']},
    ]

    optimal_params = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='roc_auc',  # NOTE: The default value for scoring results in worse performance...
        # For more scoring metics see:
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        verbose=0  # If you want to see what Grid Search is doing, set verbose=2
    )

    optimal_params.fit(X_train_scaled, y_train)
    C, gamma = optimal_params.best_params_['C'], optimal_params.best_params_['gamma']
    clf_svm = SVC(random_state=42, C=C, gamma=gamma)
    clf_svm.fit(X_train_scaled, y_train)
    dump(clf_svm, './model/svm_model.joblib')  # save model
    ConfusionMatrixDisplay.from_estimator(clf_svm,
                                          X_test_scaled,
                                          y_test,
                                          display_labels=["Move", "Stop"])
    y_pred = clf_svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    plt.savefig("./pic/confusion_matrix.png")
    plt.clf()

    # PCA
    pca = PCA()  # NOTE: By default, PCA() centers the data, but does not scale it.
    X_train_pca = pca.fit_transform(X_train_scaled)

    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = [str(x) for x in range(1, len(per_var) + 1)]

    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('Scree Plot')
    plt.savefig("./pic/pca_scree_plot.png")
    train_pc1_coords = X_train_pca[:, 0]
    train_pc2_coords = X_train_pca[:, 1]

    # NOTE:
    # pc1 contains the x-axis coordinates of the data after PCA
    # pc2 contains the y-axis coordinates of the data after PCA

    # center and scale the PCs...
    pca_train_scaled = preprocessing.scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

    num_features = np.size(pca_train_scaled, axis=1)
    param_grid = [
        {'C': [1, 10, 100, 1000],
         'gamma': [1 / num_features, 1, 0.1, 0.01, 0.001, 0.0001],
         'kernel': ['rbf']},
    ]

    optimal_params = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring='roc_auc',  # NOTE: The default value for scoring results in worse performance...
        # For more scoring metics see:
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        verbose=0  # NOTE: If you want to see what Grid Search is doing, set verbose=2
    )

    optimal_params.fit(pca_train_scaled, y_train)
    C, gamma = optimal_params.best_params_['C'], optimal_params.best_params_['gamma']
    clf_svm = SVC(random_state=42, C=C, gamma=gamma)
    clf_svm.fit(pca_train_scaled, y_train)

    # Transform the test dataset with the PCA...
    X_test_pca = pca.transform(X_test_scaled)
    test_pc1_coords = X_test_pca[:, 0]
    test_pc2_coords = X_test_pca[:, 1]

    x_min = test_pc1_coords.min() - 1
    x_max = test_pc1_coords.max() + 1

    y_min = test_pc2_coords.min() - 1
    y_max = test_pc2_coords.max() + 1

    xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                         np.arange(start=y_min, stop=y_max, step=0.1))

    Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contourf(xx, yy, Z, alpha=0.1)

    # now create custom colors for the actual data points
    cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])

    scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_test,
                         cmap=cmap,
                         s=100,
                         edgecolors='k',  # 'k' = black
                         alpha=0.7)

    # now create a legend
    legend = ax.legend(scatter.legend_elements()[0],
                       scatter.legend_elements()[1],
                       loc="upper right")
    legend.get_texts()[0].set_text("Move")
    legend.get_texts()[1].set_text("Stop")

    # now add axis labels and titles
    ax.set_ylabel('PC2')
    ax.set_xlabel('PC1')
    ax.set_title('Decison surface using the PCA transformed/projected features')
    plt.savefig('./pic/svm.png')

    return f"Complete training. Accuracy: {accuracy:.2f}"

