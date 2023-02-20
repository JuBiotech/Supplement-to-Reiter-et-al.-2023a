
####################### Packages #######################
# Common
import csv
import io
import itertools
import math
import os
import random
import re
import subprocess
import json
import warnings
from abc import ABCMeta
from copy import deepcopy
from inspect import isclass
from pathlib import Path
from sys import stdout

warnings.filterwarnings("ignore")
import pandas

pandas.set_option('display.max_colwidth', -1)
import numpy

import seaborn

FLOAT_DTYPES = (numpy.float64, numpy.float32, numpy.float16)
numpy.set_printoptions(threshold=1000)
import scipy
from scipy import odr
#from IPython.core.display import HTML, display
from statsmodels.stats.multicomp import MultiComparison, pairwise_tukeyhsd

#display(HTML("<style>.container { width:100% !important; }</style>"))
#display(HTML("<style>.output_result { max-width:100% !important; }</style>"))
import adjustText
import matplotlib
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import networkx as nx
import seaborn as sns
import sklearn
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse, Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import cross_decomposition, decomposition, metrics, model_selection, pipeline, preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA as skPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs import incr_mean_variance_axis, inplace_column_scale, mean_variance_axis
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

matplotlib.rcParams['axes.formatter.limits'] = [-20,20]

# Font
text_style = 'Arial'
bold = True
if text_style == 'Arial':
    if bold:
        font = {
            'family':'sans-serif',
            'sans-serif':'Arial',
            'weight':'bold'
            }
    else:
        font = {
            'family':'sans-serif',
            'sans-serif':'Arial'
            }
else:
    if bold:
        font = {
            'family':'serif',
            'serif':'Times New Roman', 
            'weight':'bold'
            }
    else:
        font = {
            'family':'serif',
            'serif':'Times New Roman', 
            'weight' : 'normal'
            }

rc('font',**font)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
####################### Functions #######################
def warn(*args, **kwargs):
    pass

def create_folder(path_results, foldername):
    """
    Create folder.

    Create a folder at path location and name it.

    Parameters
    ----------
    path_results : path
        Pathlib path for folder creation.
    foldername : str
        Name of the new folder.

    Returns
    -------
    path_folder : path
        Pathlib path of new folder.
    """
    path_folder = path_results.joinpath(foldername)
    path_folder.mkdir(exist_ok = True)
    return path_folder

def multiple_testing_correction(pvalues, correction_type="Holm-Bonferroni"):
    """
    Consistent with R - print
    correct_pvalues_for_multiple_testing([0.0, 0.01, 0.029, 0.03, 0.031, 0.05,
                                          0.069, 0.07, 0.071, 0.09, 0.1])
    """
    pvalues = numpy.array(pvalues)
    sample_size = pvalues.shape[0]
    qvalues = numpy.empty(sample_size)
    if correction_type == "Bonferroni":
        # Bonferroni correction
        qvalues = sample_size * pvalues
    elif correction_type == "Holm-Bonferroni":
        # Bonferroni-Holm correction
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            qvalues[i] = (sample_size-rank) * pvalue
    elif correction_type == "FDR":
        # Benjamini-Hochberg, AKA - FDR test
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = sample_size - i
            pvalue, index = vals
            new_values.append((sample_size/rank) * pvalue)
        for i in range(0, int(sample_size)-1):
            if new_values[i] < new_values[i+1]:
                new_values[i+1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            qvalues[index] = new_values[i]
    qvalues[qvalues>1] = 1
    return qvalues

def save_df(df, path_folder, filename:str, index:False):
    """
    Save dataframe.

    Save dataframe at given path with filename in xlsx format.

    Parameters
    ----------
    df : dataframe
        Dataframe to save.
    path_folder : path
        Pathlib path to save dataframe.
    filename : str
        Name of file.
    index : bool, default 'False'
        Transfer index to xlsx file.
    
    Returns
    -------
    path_file : path
        Path to file.
    """
    path_file = path_folder.joinpath(f'{filename}.xlsx')
    df.to_excel(path_file, index = index)
    return path_file

def save_dict(dictionary, path_folder, filename:str, single_files:bool, index=False):
    """
    Save dictionary.

    Save dictionary at given path with filename in xlsx format.

    Parameters
    ----------
    dictionary : dict
        Dictionary to save.
    path_folder : path
        Pathlib path to save dictionary.
    filename : str
        Name of file.
    single_files : bool
        Save dictionary entries as single xlsx files.
    index : bool, default 'False'
        Transfer index to xlsx file.
    
    Returns
    -------
    path_file : path
        Path to file (if single_files == False).
    """
    if single_files == False:
        path_file = path_folder.joinpath(f'{filename}.xlsx')
        writer = pandas.ExcelWriter(path_file)
        for key in dictionary:
            dictionary[key].to_excel(writer, sheet_name = f'{key}', index=index)
        writer.save()
        return path_file
    else:
        for key in dictionary:
            path_file = path_folder.joinpath(f'{key}.xlsx')
            dictionary[key].to_excel(path_file, index=index)
        return
    
def save_list(list_text, path_folder, filename:str):
    """
    Save list.

    Save list at given path with filename in csv format.

    Parameters
    ----------
    list_text : dict
        Dictionary to save.
    path_folder : path
        Pathlib path to save list.
    filename : str
        Name of file.
    
    Returns
    -------
    path_file : path
        Path to file.
    """
    path_file = path_folder.joinpath(f'{filename}.txt')
    with open(path_file, "w") as outfile:
        for entries in list_text:
            outfile.write(entries)
            outfile.write("\n")
    return path_file

def create_palette(k, reverse = False):
    """
    Create palette.

    Create palette with corporate design.

    Parameters
    ----------
    k : int
        Number of colors.
    reverse : bool, default 'False'
        Reverse the palette order.
    
    Returns
    -------
    palette : list
        Color palette list.
    """
    palette = seaborn.cubehelix_palette(n_colors = k, start=2.2, rot=0.2,gamma=2.1, hue=1, light=0.9, dark=0.4, reverse=reverse, as_cmap=False)
    return palette

def split_list(iterable, splitters):
    """
    Split list.

    Split list at given splitter.

    Parameters
    ----------
    iterable : list
        List to split.
    splitters : list
        List with splitters.
    
    Returns
    -------
    split_lists : list
        Splitted list.
    """
    # Find index of each splitter value in the list
    indexes = []
    for splitter in sorted(splitters):
        try:
            split = iterable.index(splitter)
            indexes.append(split)
        except ValueError:
            # Splitter not found in list
            pass
    # Split the iterable into sublists based on indices
    split_lists = []
    start = 0
    for index in sorted(indexes):
        split_lists.append(iterable[start:index])
        start = index
    split_lists.append(iterable[start:])
    return split_lists

def get_combinations(the_set):
    """
    Get all combinations.

    Get all combinations without redraw.

    Parameters
    ----------
    the_set : set
        Set of labels.

    Returns
    -------
    list_combinations : list
        List of combinations.  
    """
    list_combinations = list(itertools.combinations(the_set, r=2))
    return list_combinations

def get_product_combinations(the_set):
    """
    Get all combinations.

    Get all combinations with redraw.

    Parameters
    ----------
    the_set : set
        Set of labels.

    Returns
    -------
    list_product : list
        List of combinations.  
    """
    list_product = list(itertools.product(the_set, the_set))
    return list_product

def get_product_combinations_two_sets(set_a, set_b):
    list_product = list(itertools.product(set_a, set_b))
    return list_product

def splitDataFrameList(df, target_column, separator):
    '''
    Split strings by seperator in dataframe.

    Applies split to column of dataframe for separator.
    Splittet element will be appended to dataframe.

    Parameters
    ----------
    df : dataframe
        Dataframe to split
    target_column : str
        Column with concatenated strings
    separator : str
        Separator like -, / etc.

    Returns
    -------
    df_new : dataframe
        A dataframe with each entry for the target column separated, with each element moved into a new row. 
        The values in the other columns are duplicated across the newly divided rows.
    '''
    row_accumulator = []

    def splitListToRows(row, separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    df.apply(splitListToRows, axis=1, args = (separator, ))
    df_new = pandas.DataFrame(row_accumulator)
    return df_new

def markerselection(n_marker):
    '''
    Get list of markers.

    Create a list of different markers for plots.

    Parameters
    ----------
    n_marker : int
        Number of markers needed.

    Returns
    -------
    list_marker_select : list
        List of markers.
    '''
    list_marker = ['o','s','P','D','X','v','p','>','h','^','*','<']
    if n_marker < len(list_marker):
        list_marker_select = list_marker[:n_marker]
    else:
        list_marker_repetition = numpy.ceil(n_marker/len(list_marker))*list_marker
        list_marker_select = list_marker_repetition[:n_marker]
    return list_marker_select

def confidence_ellipse_group(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    '''
    Add confidence ellipse.

    Calculate confidence ellipse of datapoints and add to plot.

    Parameters
    ----------
    x : array
        x-Variable.
    y : array
        y-Variable.
    ax : object
        Axes.
    n_std : float
        Standard deviation for confidence ellipse.
    facecolor : str
        Facecolor of ellipse.
    kwargs : dict
        Kwargs.

    Returns
    -------
    ax : object
        Axes.
    '''
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = numpy.cov(x, y)
    pearson = cov[0, 1]/numpy.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = numpy.sqrt(1 + pearson)
    ell_radius_y = numpy.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = numpy.sqrt(cov[0, 0]) * n_std
    mean_x = numpy.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = numpy.sqrt(cov[1, 1]) * n_std
    mean_y = numpy.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)