import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def VisualizeNetwork(ax, NodeCoordinates, NodalValues, EdgeConnections, FlowValues, color='b', height=0):
    """
    Visualizes a network using circles and edges

    EXAMPLE 1
    clear all; close all
    NodeCoordinates = [0 0; 3 4; 0 7] 
    NodalValues     = [ 1;  0.5; 1.5] 
    EdgeConnections = [1 2; 2 3; 3 1]
    FlowValues      = [10;   3;  20]  
    color           = 'r';
    height          = 0.5;
    VisualizeNetwork(NodeCoordinates,NodalValues,EdgeConnections,FlowValues,height);
    axis equal
    
    EXAMPLE 2
    VisualizeNetwork([0 0; 3 4],[1; 0.5], [1 2],[10])
    """
    points = 20

    for i in range(len(NodalValues)):
        center = [NodeCoordinates[i, 0], NodeCoordinates[i, 1]]
        radius = NodalValues[i]
        DrawCircle(ax, center, radius, color, height, points)

    for i in range(len(FlowValues)):
        nodeA = [NodeCoordinates[EdgeConnections[i, 0]-1, 0], NodeCoordinates[EdgeConnections[i, 0]-1, 1]]
        nodeB = [NodeCoordinates[EdgeConnections[i, 1]-1, 0], NodeCoordinates[EdgeConnections[i, 1]-1, 1]]
        thickness = max(0.1, abs(FlowValues[i]))
        linestyle = '-' if FlowValues[i] >= 0 else '--'
        DrawEdge(ax, nodeA, nodeB, thickness, color, height, linestyle)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')

# ***************************************************
def DrawCircle(ax, center, radius, color, height, points=20):
    """
    Draws a circle

    EXAMPLE 1
    center = [-3 7];
    radius = 4;
    color = 'r';
    height = 0;
    points = 20;
    DrawCircle(ax, center, radius, color, height, points);

    EXAMPLE 2
    DrawCircle(ax, [-3 7], 4);

    Copyright Luca Daniel, MIT 2013; TA staff 2023
    """

    t = np.linspace(0, 2 * np.pi, points)
    x = np.sin(t) * radius + center[0]
    y = np.cos(t) * radius + center[1]
    z = np.zeros_like(x) + height

    verts = [list(zip(x, y, z))]
    ax.add_collection3d(Poly3DCollection(verts, color=color))

# ***************************************************
def DrawEdge(ax, nodeA, nodeB, thickness, color, height, linestyle='-'):
    """
    Draws an edge between two nodes 

    EXAMPLE 1
    nodeA = [0 0 0];
    nodeB = [-3 7 0];
    color = 'r';
    thickness = 3;
    DrawEdge(ax, nodeA,nodeB,thickness,color);

    EXAMPLE 2
    DrawEdge(ax, [0 0],[-3 7]);

    Copyright Luca Daniel, MIT 2013; TA staff, MIT 2023
    """

    X = [nodeA[0], nodeB[0]]
    Y = [nodeA[1], nodeB[1]]
    Z = [height, height]

    ax.plot(X, Y, Z, linewidth=thickness, color=color, linestyle=linestyle)
