import matplotlib.pyplot as plt
import numpy as np
import time

def nnPlot(nn, ax):
    layers = [nn.inputLayerID] + nn.hiddenLayerIDs + [nn.outputLayerID]
    x, y = [], []
    ax.clear()
    for layerID in layers:
        for nodeID in range(nn.layers[layerID].layerSize):
            x.append(layerID)
            y.append(nodeID)
            for connectedLayer in nn.layers[layerID].layerConnectedTo:
                connectedLayerID = connectedLayer.layerID
                for endNodeID in nn.layers[layerID].node[nodeID].theta[connectedLayerID].keys():
                    w = nn.layers[layerID].node[nodeID].theta[connectedLayerID][endNodeID]
                    ax.plot([layerID, connectedLayerID], [nodeID, endNodeID], 'b-', LineWidth=w)
    ax.plot(x, y, "ro", markersize=15)


class Trainer(object):
    def __init__(self, neuralN, alpha, lamda, view=False, verify=False):
        self.NN = neuralN
        self.NN.verify = verify
        self.NN.learningRate = alpha
        self.NN.regularizationFactor = lamda
        self.view = False
        self.trainingCost = []
        self.testingCost = []
        if view:
            self.view = True
            plt.ion()
            plt.pause(0.001)
        layers = [self.NN.inputLayerID] + self.NN.hiddenLayerIDs + [self.NN.outputLayerID]
        mLayerSize = 0
        for id_ in layers:
            mLayerSize = max(mLayerSize, self.NN.layers[id_].layerSize)

        self.costFig = plt.figure("Cost function vs Epoch")
        self.costAx = self.costFig.gca()

        self.nnFig = plt.figure("NN model")
        self.nnAx = self.nnFig.gca()
        self.nnAx.axis([-1, len(layers), mLayerSize, -1])

    def train(self, stepPercentage, iterationsPerStep, runtimes):
        print("Training started with {}:{} training,testing ratio and {} iterations per training set.".format(
            100 - stepPercentage, stepPercentage, iterationsPerStep))

        step = round(len(self.NN.X) * stepPercentage / 100)

        trainX = np.append(self.NN.X[0:step], self.NN.X[step:]).reshape(-1, self.NN.noOfFeatures)
        trainY = np.append(self.NN.y[0:step], self.NN.y[step:]).reshape(len(trainX), -1)
        testX = self.NN.X[step:2*step]
        testY = self.NN.y[step:2*step]

        iterations = iterationsPerStep

        time_avg = 0
        for i in range(10):
            time_temp = time.time()
            self.NN.learn(trainX, trainY)
            J_train = self.NN.loss(trainY)
            self.NN.forward_prop(testX, testY, back_prop=False)
            J_test = self.NN.loss(testY)
            time_avg += time.time() - time_temp
        time_avg /= 10
        print("average time per iteration", time_avg, ". ETA", time_avg * iterations *len(self.NN.X)//step, "per runtime")
        print("Total ETA", time_avg* runtimes * iterations*len(self.NN.X)//step)

        for run in range(runtimes):
            print("Runtime {}/{}".format(run+1, runtimes))
            start = 0
            while start < len(self.NN.X):
                print(start)
                trainX = np.append(self.NN.X[0:start], self.NN.X[start + step:]).reshape(-1, self.NN.noOfFeatures)
                trainY = np.append(self.NN.y[0:start], self.NN.y[start + step:]).reshape(len(trainX), -1)
                testX = self.NN.X[start:start + step]
                testY = self.NN.y[start:start + step]
                for i in range(iterations):
                    if i % (iterations // (iterations//20)) == 0 and self.view:
                        nnPlot(self.NN, self.nnAx)
                        self.costAx.plot(self.trainingCost, 'b')
                        self.costAx.plot(self.testingCost, 'r')
                        plt.pause(0.0000001)
                    self.NN.learn(trainX, trainY)
                    J_train = self.NN.loss(trainY)
                    self.NN.forward_prop(testX, testY, back_prop=False)
                    J_test = self.NN.loss(testY)
                    self.trainingCost.append(J_train)
                    self.testingCost.append(J_test)

                start += step

        if plt.isinteractive():
            plt.ioff()

        return self.trainingCost, self.testingCost

    def trainWithoutTesting(self, iterations):
        self.trainingCost = []

        trainX = self.NN.X
        trainY = self.NN.y
        time_avg = 0
        for i in range(10):
            time_temp = time.time()
            self.NN.learn(trainX, trainY)
            J_train = self.NN.loss(trainY)
            self.trainingCost.append(J_train)
            time_avg += time.time() - time_temp
        time_avg /= 10
        print("average time per iteration", time_avg,". ETA", time_avg*(iterations-10))

        for i in range(10, iterations):
            if i % (iterations // 10) == 0:
                print(i)
            if i % (iterations // 10) == 0 and self.view:
                nnPlot(self.NN, self.nnAx)
                self.costAx.plot(self.trainingCost, 'b')
                plt.pause(0.000001)
            self.NN.learn(trainX, trainY)
            J_train = self.NN.loss(trainY)
            self.trainingCost.append(J_train)

        plt.ioff()

        return self.trainingCost

    def analyzeLearningRate(self):
        tempLR = self.NN.learningRate
        self.NN.learningRate = 0.001
        temp = []
        for i in range(5):
            temp.append(self.trainWithoutTesting(100))
            self.NN.learningRate *= 10
        self.NN.learningRate = tempLR

        plt.figure("Learning rate analysis")
        plt.title("Learning rate analysis")
        plt.ion()
        for i in range(len(temp)):
            plt.plot(temp[i])
        plt.ioff()
        plt.legend(["0.001", "0.01", "0.1", "1", "10"])


    def plotCost(self):
        nnPlot(self.NN, self.nnAx)
        self.costAx.plot(self.trainingCost, 'b')
        self.costAx.plot(self.testingCost, 'r')
        plt.pause(0.001)

    def plotThetaHistory(self):
        f = plt.figure("Theta vs Epoch")
        plt.title("Theta Flow")
        plt.ion()
        for layer1ID in self.NN.hiddenLayerIDs + [self.NN.outputLayerID]:
            layer1 = self.NN.layers[layer1ID]
            for layer2 in layer1.layerConnectedFrom:
                layer2ID = layer2.layerID

                for node1ID in layer1.node.keys():
                    node1 = layer1.node[node1ID]
                    for node2ID in layer2.node.keys():
                        try:
                            plt.plot(node1.thetaHistory[layer2ID][node2ID])
                        except Exception as e:
                            pass
        plt.ioff()

    def contourAnd3dPlot(self):
        """
        IFF 2 input features.
        """
        if self.NN.noOfFeatures != 2:
            print("Cannot draw if the number of features are not 2.")
            return

        x1 = np.linspace(self.NN.XMin[0], self.NN.XMax[0], 100)
        x2 = np.linspace(self.NN.XMin[1], self.NN.XMax[1], 100)

        # Create 2-d versions of input for plotting
        a, b = np.meshgrid(x1, x2)

        # Join into a single input matrix:
        allInputs = np.zeros((a.size, 2))
        allInputs[:, 0] = a.ravel()
        allInputs[:, 1] = b.ravel()

        allInputs, inputMean, inputMax, inputMin = self.NN.normalize(allInputs)
        allOutputs = self.NN.forward_prop(allInputs)

        # Contour Plot:
        yy = np.dot(x1.reshape(100, 1), np.ones((1, 100)))
        xx = np.dot(x2.reshape(100, 1), np.ones((1, 100))).T

        if self.NN.typeOfNN == "Classification":
            newAllOutputs = []
            for i in range(len(allOutputs)):
                newAllOutputs.append(0)
                for j in range(len(allOutputs[i])):
                    newAllOutputs[-1] += (j+1)*allOutputs[i][j]
            allOutputs = np.array(newAllOutputs)
        try:
            plt.figure("Contour plot")
            plt.title("Contour plot")
            if self.NN.typeOfNN == "Classification":
                CS = plt.contour(xx, yy, allOutputs.reshape(100, 100))
                plt.ion()
                x_rescale = (self.NN.XMax - self.NN.XMin) + self.NN.XMin[0] + self.NN.XMean
                for i in range(len(self.NN.X)):
                    temp = list(self.NN.y[i]).index(1)
                    if temp == 0:
                        clr = (0,0,1)
                    elif temp == 1:
                        clr = (0,1,0)
                    elif temp == 2:
                        clr = (1, 0, 0)
                    elif temp == 3:
                        clr = (1,1,0)
                    else:
                        print("Please define a color")
                        clr = (0,0,0)
                    plt.plot(self.NN.X[i][0]*x_rescale[0], self.NN.X[i][1]*x_rescale[1], "o", markersize=10, color=clr)
                plt.ioff()
            else:
                CS = plt.contour(xx, yy, (allOutputs * (self.NN.yMax - self.NN.yMin) + self.NN.yMin[0] + self.NN.yMean).reshape(100, 100))
            plt.clabel(CS, inline=1, fontsize=10)

        except Exception as e:
            # probably because all the outputs are constant
            print("Contour plot failed:", e)

        # 3d plot
        from mpl_toolkits.mplot3d import Axes3D
        try:
            fig = plt.figure("3D plot")
            plt.title("3D plot")
            ax = fig.gca(projection='3d')
            if self.NN.typeOfNN == "Classification":
                temp_y = []
                for i in range(len(self.NN.y)):
                    for j in range(len(self.NN.y[i])):
                        if self.NN.y[i][j] == 1:
                            temp_y.append(j+1)
                ax.scatter(self.NN.X[:, 0], self.NN.X[:, 1], temp_y, c='k', alpha=1, s=30)
                surf = ax.plot_surface(xx, yy, allOutputs.reshape(100, 100) )

            else:
                x_rescale = (self.NN.XMax - self.NN.XMin) + self.NN.XMin[0] + self.NN.XMean
                ax.scatter(self.NN.X[:, 0]*x_rescale[0], self.NN.X[:, 1]*x_rescale[1], self.NN.y * (self.NN.yMax - self.NN.yMin) + self.NN.yMin[0] + self.NN.yMean, c='k', alpha=1, s=30)
                surf = ax.plot_surface(xx, yy, allOutputs.reshape(100, 100) * (self.NN.yMax - self.NN.yMin) + self.NN.yMin[0] + self.NN.yMean)

        except Exception as e:
            print("3D plot failed:", e)