import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Graph:
    def __init__(self):
        self.color_map = plt.get_cmap('viridis')

    def scatter_2Ddata(self, x_train, y_train, x_test=None, y_test=None):
        """
        Scatters a 2D data. Note that if you haven't implemented a train_test split, you don't need to fill x_test and y_test parameters
        :param x_train: the x_training value
        :param y_train: the y_training value
        :param x_test: the x_testing value (Default=None)
        :param y_test: the y_testing value (Default=None)
        :return: None
        """
        plt.scatter(x_test, y_test, color=self.color_map(0.9), s=10)
        #if (x_train != None and y_train != None):
        plt.scatter(x_train, y_train, color=self.color_map(0.5), s=10)


    def graph_2D(self, X, y, title=None, xlabel=None, ylabel=None):
        """
        Graphs a 2D equation
        :param X: a line space, or a range of values
        :param y: The output of the line space/range of values
        :param title: Title of the plot
        :param xlabel: The X-label of the plot
        :param ylabel: The Y-label of the plot
        :return: None
        """
        plt.plot(X, y, color='black')
        if (title!=None):
            plt.title(title)
        if (xlabel!=None):
            plt.xlabel(xlabel)
        if (ylabel!=None):
            plt.ylabel(ylabel)
        plt.show()

    def graph_3D(self, x1, x2, Y, wireframe=True):
        """
        Graphs a 3D equation in a 3D space
        :param x1: the first feature value (linspace/range) (x)
        :param x2: the second feature value (linspace/range) (y)
        :param Y: the output value (array) (z)
        :param wireframe: selects that the graph would be wireframe of not (bool)
        :return: None
        """
        ax = plt.axes(projection='3d')
        if wireframe:
            ax.plot_wireframe(x1, x2, Y)
        else:
            ax.plot_surface(x1, x2, Y)
        ax.set_xlabel('1st Feature')
        ax.set_ylabel('2nd Feature')
        ax.set_zlabel('Label')
        plt.show()

    def scatter_3Ddata(self, x1, x2, y):
        """
        Scatters a 3D data in a 3D space
        :param x1: The x-coordinates of a 3-D data (array)
        :param x2: The y-coordinates of a 3-D data (array)
        :param y: The z-coordinates of a 3-D data (array)
        :return: None
        """
        ax = plt.axes(projection='3d')
        ax.scatter(x1, x2, y, marker='o', color=self.color_map(0.9))
