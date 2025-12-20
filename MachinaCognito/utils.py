import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap='bone'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
        
    fig, ax = plt.subplots(1, 1)
    
    print(cm)
    sc = ax.imshow(cm, interpolation='nearest', cmap=cmap, alpha=0.8)
    ax.set_title(title)
    plt.colorbar(sc)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes, rotation=45)
    ax.set_yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i,j ], fmt), 
                    horizontalalignment='center', verticalalignment='center',
                    color='xkcd:ocean' if i == j else 'xkcd:twilight', fontsize=30)
            
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.show()


def plot_learning_curve(estimator, title, X, y,
                       ylim=None, cv=5, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5),
                       scoring='accuracy'):
    
    """
    Generate a plot of the test and training learning curve for a given model (estimator).
    
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type is cloned for each validation
        
    title : string
        Title for the plot
        
    X : array-like, (n_samples, n_features)
        Training vector where n_samples is the number of samples, and n_features
        is the number of features.
        
    y : array-like, (n_samples) || (n_samples, n_features), optional
        Target relative to X for classification or regression.
        Set to None for unsupervised learning
        
    ylim : tuple, (ymin, ymax), optional
        Define minimum and maximum for the y-axis of the plot
        
    cv : int, cross-validation generator or iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs:
        - None, uses default 3-fold cross-validation
        - integer, specifies the number of folds
        - :term:'CV splitter'
        - iterable yielding (train, test) splits as arrays of indicies
        
        For integer/None inputs, if "y" is binary or multiclass, :class:'StratifiedKFold' is
        used. If the estimator is not a classifier, or if "y" is neither binary or multiclass,
        then :class:"KFold" is used.
        
    n_jobs : int || None, optional (default=1)
        Number of jobs to run in parallel
        "1" means that 1 job will be run
        "-1" means using all processors
        
    train_sizes: array-like, (n_ticks,), dtype float or int
        Relative or absolute number of training examples that will be used to generate the learning
        curve. If the dytpe is float, it is regarded as a fraction of the maximum size of the training
        set. Otherwise it is the absolute size of the training sets. Note that for classification
        the number of samples usually has to be big enough to contain at least one sample from each
        class.
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                           train_sizes=train_sizes, scoring=scoring)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, color='xkcd:ocean', alpha=0.5)
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                   test_scores_mean + test_scores_std, color='xkcd:pumpkin', alpha=0.5)
    
    ax.plot(train_sizes, train_scores_mean, 'o-', color='xkcd:ocean', alpha=0.8,
           label='Training Score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='xkcd:pumpkin', alpha=0.8,
           label='Test Score (Cross-Validation)')
    
    ax.legend(loc='best')
    ax.set_title(title)
    ax.grid()
    ax.set_xlabel('Training Examples')
    ax.set_ylabel(str(scoring))
    
    if ylim is not None:
        ax.set_ylim(*ylim)
        
    plt.show()