'''
coordinates.py
natural-mixer

2012 Brandon Mechtley
Arizona State University

Various tools for working with barycentric coordinates of arbitrary dimension including routines
for display. All routines automatically normalize barycentric coordinates.
'''

from itertools import izip
import numpy as np
import matplotlib.pyplot as pp

def project_pointline(p, a, b):
    '''Euclidean projection of a point onto a line segment.
        p: np.ndarray
            point of arbitrary dimensionality.
        a: np.ndarray
            first endpoint of the line segment.
        b: np.ndarray
            second endpoint of the line segment.'''
    
    return a + (np.dot(p - a, b - a) / np.dot(b - a, b - a)) * (b - a)

def bary2cart(bary, corners=None):
    '''Convert barycentric coordinates to cartesian coordinates given the cartesian coordinates of 
    the corners.
        bary: np.ndarray
            barycentric coordinates to convert. If this matrix has multiple rows, each row is
            interpreted as an individual coordinate to convert.
        corners: np.ndarray
            cartesian coordinates of the corners.'''
    
    if corners == None:
        corners = polycorners(bary.shape[-1])
    
    if len(bary.shape) > 1:
        return np.array([np.sum(b / np.sum(b) * corners.T, axis=1) for b in bary])
    else:
        return np.sum(bary / np.sum(bary) * corners.T, axis=1)

def lattice(ncorners=3, sides=False):
    '''Create a lattice of linear combinations of barycentric coordinates with ncorners corners.
    This lattice is constructed from the corners, the center point between them, points between the
    corners and the center, and pairwise combinations of the corners and the center.
        ncorners: int, optional
            number of corners of the boundary polygon (default 3).
        sides: bool, optional
            whether or not to include pairwise combinations of the corners (i.e. sides) in the
            lattice (default False).'''
    
    # 1. Corners.
    coords = list(np.identity(ncorners))
    
    # 2. Center.
    center = np.array([1. / ncorners] * ncorners)
    coords.append(center)
    
    # 3. Corner - Center.
    for i in range(ncorners):
        for j in range(i + 1, ncorners):
            coords.append((coords[i] + coords[j] + center) / 3)
    
    # 4. Corner - Corner - Center.
    for i in range(ncorners):
        coords.append((coords[i] + center) / 2)
    
    # 5. Corner - Corner (Sides)
    if sides:
        for i in range(ncorners):
            for j in range(i + 1, ncorners):
                coords.append((coords[i] + coords[j]) / 2)
    
    # 5. Return unique coordinates (some duplicates using this method with e.g. ncorners=2)
    return np.array(list(set(tuple(c) for c in coords)))

def polycorners(ncorners=3):
    '''Return 2D cartesian coordinates of a regular convex polygon of a specified number of
    corners.
        ncorners: int, optional
            number of corners for the polygon (default 3).'''
    
    center = np.array([0.5, 0.5])
    points = []
    
    for i in range(ncorners):
        angle = (float(i) / ncorners) * (np.pi * 2) + (np.pi / 2)
        x = center[0] + np.cos(angle) * 0.5
        y = center[1] + np.sin(angle) * 0.5
        points.append(np.array([x, y]))
    
    return np.array(points)

def verttext(pt, txt, center=[.5,.5], dist=1./15, color='red'):
    '''Display a text label for a vertex with respect to the center. The text will be a certain
    distance from the specified vertex in the direction of the vector extending from the center
    point.
        pt: np.ndarray
            two-dimensional array of cartesian coordinates of the point to label.
        txt: str
            text to display. Text will be horizontally and vertically centered around pt.
        center: np.ndarray, optional
            reference center point used for arranging the text around pt (default [.5, .5]).
        dist: float, optional
            distance between point and the text (default 1./15).
        color: str, optional
            matplotlib color of the text (default 'red').'''
    
    vert = pt - center
    s = np.sum(np.abs(vert))
    
    if s == 0:
        vert = np.array([0., 1.])
    else:
        vert /= s
    
    vert *= dist
    
    pp.text(
        pt[0] + vert[0],
        pt[1] + vert[1],
        txt,
        horizontalalignment='center',
        verticalalignment='center',
        color=color
    )

def polyshow(coords, color=None, label=None, labelvertices=False, polycolor=None, lines=[]):
    '''Plot a regular convex polygon surrounding one or more barycentric coordinates within the 
    it. Vertices and corners will be labeled sequentially starting at 0.
        coords: np.ndarray or list
            one or more barycentric coordinates of equal arbitrary dimension. The dimensionality of 
            the coordinates will correspond to the number of vertices of the polygon that is drawn.
        color: str or list, optional
             color in which to draw the coords. If color is a list of the same length as coords, 
             each entry will correspond to the respective coordinates.'''
    
    
    # Defaults.
    coords = np.array(coords)
    if len(coords.shape) < 2: coords = [coords]
    for coord in coords:
        if np.sum(coord) > 0:
            coord /= np.sum(coord)
    
    if color == None: color = 'blue'
    if type(color) == str: color = [color] * len(coords)
    
    if label == None: label = ''
    if type(label) == str: label = [label] * len(coords)
    
    # Number of sides.
    d = len(coords[0])
    
    # Cartesian coordinates of the vertices of the polygon and each point.
    corners = polycorners(d)
    cart = [np.sum([c * cnr for c, cnr in izip(coord, corners)], axis=0) for coord in coords]
    cart = np.array(cart)
    
    # Figure/axes setup.
    f = pp.figure(frameon=False)
    ax = pp.axes(frameon=False)
    ax.axis('equal')
    pp.xticks([])
    pp.yticks([])
    
    # Add the polygon and its vertices to the figure.
    ax.add_patch(pp.Polygon(corners, closed=True, fill=False, alpha=0.5))
    ax.scatter(corners[:,0], corners[:,1], color='red', s=50)
    if labelvertices:
        map(lambda i: verttext(corners[i], '$v_%d$' % i), range(len(corners)))
    
    # Add any extra lines to the figure.
    map(ax.add_line, lines)
    
    # Add the interior points and their labels.
    ax.scatter(cart[:,0], cart[:,1], color=color, s=100)
    
    for c, txt, clr in izip(cart, label, color):
        verttext(c, txt, color=clr)
    
    return f

def baryedges(coords):
    '''Return an array of barycentric coordinates corresponding to the closest point on each edge 
    of the respective polygon.
        coords: np.ndarray
            input coordinates.'''
    
    # There ought to be an easier way to do this without switching out of barycentric coordinates,
    # but after obsessing over it for an entire day, I'll have to work on that later.
    
    d = len(coords)                     # number of edges.
    corners = polycorners(d)            # cartesian corners of the polygon.
    cart = bary2cart(coords, corners)   # cartesian coordinates of the input point.
    
    e = np.zeros((d, d))                # final bary. coords. for projection to each side.
    
    # Project the point onto each edge in cartesian space, put it back into bary. coords.
    for i1, i2 in izip(np.arange(d), np.roll(np.arange(d), 1)):
        proj = project_pointline(cart, corners[i1], corners[i2])
        distances = np.array([np.linalg.norm(corners[a] - proj, 2) for a in [i1, i2]])
        e[i1, (i1, i2)] = np.sum(distances) - distances
    
    return e
