import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Grapher:
    def __init__(self):
        self.ax = plt.axes(projection='3d')

    def graph2Deq(self, x, y, title='', xlabel='', ylabel=''):
        plt.plot(x, y, color='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def graph2Dscat(self, xtest, ytest, xtrain=None, ytrain=None):
        plt.scatter(xtest, ytest, color='red', s=10)
        if (xtrain != None and ytrain != None):
            plt.scatter(xtrain, ytrain, color='green', s=10)

    def graph3Deq(self, x1, x2, Y, wireframe=True):
        if wireframe:
            self.ax.plot_wireframe(x1, x2, Y)
        else:
            self.ax.plot_surface(x1, x2, Y)
        self.ax.set_xlabel('1st Feature')
        self.ax.set_ylabel('2nd Feature')
        self.ax.set_zlabel('Label')
        plt.show()

    def graph3Dscat(self, x1, x2, y):
        self.ax.scatter(x1, x2, y, marker='o', color='red')

