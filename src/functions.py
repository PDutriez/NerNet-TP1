from lib2to3.pytree import Base
from random import shuffle
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm, gaussian_kde, multivariate_normal
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelBinarizer

def pretty_class_EDA(data,name,columns,bins=10,title=None,figsize=(16,6)):
    """ Muestra la distribucion de los parametros de una clase y su correlacion """
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    pretty_hist(data,name,columns,bins=bins,title=f"{name} variables dist",figsize=figsize,plot=False)
    plt.subplot(1, 2, 2)
    pretty_corr_matrix(data,name,title=f"{name} variables corr",plot=False)
    plt.show()

def pretty_hist(data,name=None,columns=None,title=None,bins=10,figsize=(16,6),plot=True):
    if plot:
        plt.figure(figsize=figsize)
    if title:
        plt.title(title,fontsize=22)
    for c in columns:
        if name:
            mu, std = norm.fit(data[data["Class"] == name][c])
            plt.hist(data[data["Class"] == name][c],bins=bins,label=f'{c}: $\mu={np.round(mu,3)},\ \sigma={np.round(std,3)}$')
        else:
            mu, std = norm.fit(data[c])
            plt.hist(data[c],bins=bins,label=f'{c}: $\mu={np.round(mu,3)},\ \sigma={np.round(std,3)}$')
    plt.legend()
    if plot:
        plt.show()

def pretty_hist2d(data,clases,columns,bins=10,title=None,figsize=(16,6)):
    nullfmt = NullFormatter()         # no labels
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=figsize)

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    for c in clases:
        x = data[data["Class"] == c][columns[0]]
        y = data[data["Class"] == c][columns[1]]
        axScatter.scatter(x, y,label=c)

        # now determine nice limits by hand:
        axHistx.hist(x, bins=30,alpha=0.5)
        axHisty.hist(y, bins=30,alpha=0.5, orientation='horizontal')
    axScatter.set_xlim()
    axScatter.set_ylim()
    axScatter.set_xlabel(columns[0],fontsize=18)
    axScatter.set_ylabel(columns[1],fontsize=18)
    axScatter.legend()
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()

def pretty_corr_matrix(data,name=None,title=None,figsize=(16,6),plot=True):
    if plot:
        plt.figure(figsize=figsize)
    if title:
        plt.title(title,fontsize=22)
    if name:
        corr = data[data["Class"] == name].corr()
    else:
        corr = data.corr()
    im = plt.matshow(corr,fignum=False)
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns,fontsize=12)
    plt.yticks(range(len(corr.columns)), corr.columns,fontsize=12)
    if plot:
        plt.show()

def pretty_scatter(rvs,figsize=(16,6)):
    fig = plt.figure(figsize=figsize)

    x,y = generate_dataset(rvs)
    enc = OneHotEncoder(handle_unknown='ignore')
    y_cat = enc.fit_transform(y.reshape(-1,1))
    plt.scatter(x[:,0],x[:,1],c=y,s=20)

    plt.show()

def generate_dataset(random_variables):
    X = np.array([]).reshape(0, len(random_variables[0][0]))
    y = np.array([]).reshape(0, 1)
    for i, rv in enumerate(random_variables):
        X = np.vstack([X, np.random.multivariate_normal(rv[0], rv[1], rv[2])])
        y = np.vstack([y, np.ones(rv[2]).reshape(rv[2],1)*i]) 
    y = y.reshape(-1)
    return X, y

def compare_param(data,column,bins=10,figsize=(16,6),kind="kde"):
    """ Muestra la distribucion del parametro, el boxplot y la distribucion del parametro por clase """
    fig, axd = plt.subplot_mosaic([['left', 'right'],['bottom', 'bottom']], figsize=figsize,constrained_layout=True)
    # Upper Left
    mu, std = norm.fit(data[column])
    axd["left"].set_title(f"Distribución {column}",fontsize=32)
    axd["left"].hist(data[column], density=True,bins = bins,label=f'{column}: $\mu={np.round(mu,3)},\ \sigma={np.round(std,3)}$')
    mn, mx = axd["left"].set_xlim()
    axd["left"].set_xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = gaussian_kde(data[column])
    axd["left"].plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    axd["left"].set_ylabel("Densidad")
    # Upper Right
    axd["left"].set_title(f"Boxplot {column}",fontsize=32)
    axd["right"].boxplot(data[column])
    # Bottom
    clases = data["Class"].drop_duplicates().to_list()
    dic = {}
    for i,name in enumerate(clases):
        dic[name] = data[data["Class"] == name][column]
    try:
        if kind == "kde":
            pd.DataFrame(dic).plot(kind="kde",ax=axd["bottom"],title=f"Distribución de {column} KDE por clase")
        else:
            pd.DataFrame(dic).plot(kind="hist",ax=axd["bottom"],title=f"Histograma de {column} por clase",alpha=0.5,bins=bins)
    except BaseException:
        pd.DataFrame(dic).plot(kind="hist",ax=axd["bottom"],title=f"Histograma de {column} por clase",bins=bins)
    
def pretty_param(data,column,bins=10,figsize=(16,6),kind="kde"):
    """ Muestra la distribucion del parametro, el boxplot y la distribucion del parametro por clase """
    fig, axd = plt.subplot_mosaic([['left', 'right']], figsize=figsize,constrained_layout=True)
    # Upper Left
    mu, std = norm.fit(data[column])
    axd["left"].set_title(f"Distribución {column}",fontsize=32)
    axd["left"].hist(data[column], density=True,bins = bins,label=f'{column}: $\mu={np.round(mu,3)},\ \sigma={np.round(std,3)}$')
    mn, mx = axd["left"].set_xlim()
    axd["left"].set_xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = gaussian_kde(data[column])
    axd["left"].plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    axd["left"].set_ylabel("Densidad")
    # Upper Right
    axd["left"].set_title(f"Boxplot {column}",fontsize=32)
    axd["right"].boxplot(data[column])

def get_yeast_class(sample,train,print_res=True):
    clases = train["Class"].drop_duplicates().to_list()
    p = {}
    f = {}
    x_prob = 0
    for clase in clases:
        p[clase] = train[train["Class"] == clase].shape[0]/train.shape[0]
        m = train[train["Class"] == clase].mean(numeric_only=True).to_list()
        c = train[train["Class"] == clase].cov().to_numpy()
        f[clase] = multivariate_normal(m,c,allow_singular=True)
        x_prob = x_prob + f[clase].pdf(sample)*p[clase]
    
    max_prob = 0
    max_prob_class = None
    for clase in clases:
        px = f[clase].pdf(sample)*p[clase]/x_prob
        if print_res:
            print(f"P(Y={clase}|X) = {np.round(px,5)}")
        if px > max_prob:
            max_prob_class = clase
            max_prob = px
    if print_res:
        print(f"clase: {max_prob_class}, P={np.round(max_prob,5)}")

    return max_prob_class

def test_linear_regression(data,test_size,poly_ord=1):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    poly = PolynomialFeatures(poly_ord,include_bias=False)
    train_df, test_df = train_test_split(data, test_size=test_size,random_state=42)
    x_train = train_df.drop(columns="RMSD").values
    x_train = poly.fit_transform(x_train,)
    x_test = test_df.drop(columns="RMSD").values
    x_test = poly.fit_transform(x_test,)
    y_train = train_df["RMSD"].values
    y_test = test_df["RMSD"].values

    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    return x_train, y_train, x_test, y_test, y_pred

    # function for scoring roc auc score for multi-class
def multiclass_roc_auc_score(y_test, y_pred, clases,average="macro"):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(clases)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i],tpr[i],label="ROC curve (area = %0.2f)" % roc_auc[2],
)
    # lb = LabelBinarizer()
    # lb.fit(y_test)
    # y_test = lb.transform(y_test)
    # y_pred = lb.transform(y_pred)

    # for (idx, c_label) in enumerate(clases):
    #     fpr, tpr, thresholds = roc_curve(y_test[:,idx], y_pred[:,idx])
    #     plt.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    # plt.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    # plt.legend()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()

    # return roc_auc_score(y_test, y_pred, average=average)
def show_yeast_metrics(y_test,y_pred):
    score = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_weigt = f1_score(y_test, y_pred, average='weighted')
    f1_noavg = f1_score(y_test, y_pred, average=None)
    print(f"Accuracy: {np.round(score*100,3)}%")
    print(f"F1 Macro: {np.round(f1_macro*100,3)}%")
    print(f"F1 Micro: {np.round(f1_micro*100,3)}%")
    print(f"F1 Class: {np.round(f1_noavg*100,3)}%")
    return score,f1_macro,f1_micro, f1_noavg

def plot_yeast_metrics(history,figsize=(12,8)):
    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.ylabel("Loss",fontsize=18)
    plt.xlabel("Epoch",fontsize=18)
    plt.legend()
    plt.subplot(132)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.ylabel("Accuracy",fontsize=18)
    plt.xlabel("Epoch",fontsize=18)
    plt.legend()
    plt.subplot(133)
    plt.plot(history.history["auc"], label="train")
    plt.plot(history.history["val_auc"], label="val")
    plt.ylabel("AUC",fontsize=18)
    plt.xlabel("Epoch",fontsize=18)
    plt.legend()
    plt.show()

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),):
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(20, 8))

    axes[0].set_title(title,fontsize=22)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples",fontsize=12)
    axes[0].set_ylabel("MSE",fontsize=12)

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    # fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].plot(train_sizes, -train_scores_mean, "o-", color="r", label="Training error")
    axes[0].plot(train_sizes, -test_scores_mean, "o-", color="g", label="Test error")
    axes[0].legend(loc="best")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    axes[1].grid()
    axes[1].plot(fit_time_sorted, -test_scores_mean_sorted, "o-")

    axes[1].set_xlabel("fit_times",fontsize=12)
    axes[1].set_ylabel("MSE",fontsize=12)
    axes[1].set_title("Performance del modelo",fontsize=22)

    return plt