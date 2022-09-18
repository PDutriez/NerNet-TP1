from lib2to3.pytree import Base
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm, gaussian_kde, multivariate_normal
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

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
    
# rvs = []
# for clase in ["NUC","CYT","MIT"]:
#     rvs.append([df[df["Class"] == clase].mean(numeric_only=True).to_list(),df[df["Class"] == clase].cov().to_numpy(),df[df["Class"] == clase].shape[0]])

# nn.pretty_scatter(rvs)

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

def get_train_set(data):
    clases = data["Class"].drop_duplicates().to_list()
    