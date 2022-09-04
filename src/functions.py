import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm, stats
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def pretty_class_EDA(data,name,columns,bins=10,title=None,figsize=(16,6)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    pretty_hist(data,name,columns,bins=bins,title=f"{name} variables dist",figsize=figsize,plot=False)
    plt.subplot(1, 2, 2)
    pretty_corr_matrix(data,name,title=f"{name} variables corr",plot=False)
    plt.show()

def pretty_hist(data,name,columns,title=None,bins=10,figsize=(16,6),plot=True):
    if plot:
        plt.figure(figsize=figsize)
    if title:
        plt.title(title,fontsize=22)
    for c in columns:
        mu, std = norm.fit(data[data["Class"] == name][c])
        plt.hist(data[data["Class"] == name][c],bins=bins,label=f'{c}: $\mu={np.round(mu,3)},\ \sigma={np.round(std,3)}$')
    plt.legend()
    if plot:
        plt.show()

def pretty_hist2d(data,columns,bins=10,title=None,figsize=(16,6)):
    plt.figure(figsize=figsize)
    if title:
        plt.title(title,fontsize=22)
    plt.hist2d(data[columns[0]].to_numpy(),data[columns[1]].to_numpy(),bins=bins)
    plt.show()

def pretty_corr_matrix(data,name,title=None,figsize=(16,6),plot=True):
    if plot:
        plt.figure(figsize=figsize)
    if title:
        plt.title(title,fontsize=22)
    corr = data[data["Class"] == name].corr()
    im = plt.imshow(data[data["Class"] == name].corr())
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns,fontsize=12)
    plt.yticks(range(len(corr.columns)), corr.columns,fontsize=12)

    # for i in range(len(corr.columns)):
    #     for j in range(len(corr.columns)):
    #         text = plt.text(j, i, corr.to_numpy()[i, j],
    #                    ha="center", va="center", color="w")
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