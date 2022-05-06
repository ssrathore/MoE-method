from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import random

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

    Y = [ [2.9,2.9] , [2.3,0.2], [3.9,1.1], [5.4,-2.3], [3.7,3.7] ]
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    #labels = gmm.fit(X).predict(X)
    labels = [random.randint(0, 2) for x in range(np.shape(X)[0])]
    #print(labels)
    #print(X.shape)
    #print(labels)
    mrkr = ['*','^','+']
    color = ['red', 'blue', 'green']
    x = X[:, 0]
    y = X[:, 1]
    if label:
        for i in range(np.shape(X)[0]):
            ax.scatter(x[i], y[i], marker=mrkr[labels[i]],c=color[labels[i]], s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=30, zorder=2,  label='Training points')
    ax.axis('equal')
    #y1 = [2.9, 2.3, 3.8, 5.4, 8]
    #y2 = [2.9, 0.2, 1.1, -2.3, 5]
    #Y = [ [2.9,2.9] , [2.3,0.2], [3.9,1.1], [5.4,-2.3], [3.7,3.7] ]
    
    #y_pred = gmm.predict_proba(Y)
    #print(y_pred)
    #ax.scatter(y1, y2, s=40, zorder=2, c='Brown', marker='D', label='Testing points')
    #w_factor = 0.2 / gmm.weights_.max()
    #for pos, covar, w in zip(gmm.means_, gmm.covariances_ , gmm.weights_):
        #draw_ellipse(pos, covar, alpha=w * w_factor)
