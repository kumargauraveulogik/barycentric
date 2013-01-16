'''
notebookutils.py
natural-mixer

2012 Brandon Mechtley
Arizona State University

A few helpful routines for using IPython notebook.
'''

from IPython.core.pylabtools import print_figure
from IPython.core.display import display, HTML, Math, Latex
from sympy import Matrix, latex
import numpy as np


def limitprec(m, prec=3):
    '''Silly function to limit the maximum number of decimal places for a matrix of floats.
        m: np.ndarray
            input matrix'''
    
    mi = np.array(m * 10 ** prec, dtype=int)
    return np.array(mi, dtype=float) / 10 ** prec
    
def showmat(m, labels=('','',''), prec=3):
    '''Display a numpy.ndarray as a latex matrix in an IPython notebook with optional caption.
        m: np.ndarray
            array to display
        labels: (str, str, str), optional
            Latex to insert before, between, and after matrices (default ('','','')).
        prec: int, optional
            Maximum number of decimal places. Hardcoding this as opposed to using ordinary string 
            formatting because the Numpy->SymPy->IPython chain makes things confusing. Feel free to 
            propose a better method @_@'''
    
    if type(labels) != tuple and type(labels) != list:
        labels = (labels, '', '')
    elif len(labels) < 2:
        labels = (labels[0], '', '')
    elif len(labels) < 3:
        labels = (labels[0], labels[1], '')
    
    if type(m) != list and type(m) != tuple:
        m = [m]
    
    display(
        Latex(
            '%s$$' % labels[0] +
            labels[1].join([
                latex(Matrix(limitprec(a, prec)), mat_str='matrix')
                for a in m
            ]) + '$$%s' % labels[2]
        )
    )

def svgfig(figures):
    '''Display a matplotlib figure as SVG in an IPython notebook.
        f: matplotlib.figure
            figure to display as SVG
            
        Note that this routine assumes you are NOT using ipython notebook --pylab inline, as 
        otherwise the original figure will be displayed as a rasterized PNG in addition to this SVG 
        figure by default.'''
    
    if type(figures) != list and type(figures) != tuple: figures = [figures]
    
    for f in figures:
        display(HTML(print_figure(f, fmt='svg')))
