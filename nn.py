from digger.layernode import *


class NN(object):
    def __init__(self, typeOfNN, cost_function):
        self.typeOfNN = typeOfNN
        self.cost_function = cost_function
        self.X = np.array([])
        self.y = np.array([])
        self.yMean, self.yMax, self.yMin = 0, 0, 0
        self.XMean, self.XMax, self.XMin = 0, 0, 0
        self.noOfFeatures = 0
        self.noOfDataSets = 0
        self.layers = {}
        self.hiddenLayerIDs = []  # forward propagation happens according to this order
        self.inputLayerID = 0
        self.outputLayerID = 0
        self.learningRate = 1
        self.regularizationFactor = 0.000001
        self.currentEstimate = []
        self.backproplog = open('Backprop_log.txt', 'w')
        self.verify = False

    def feed(self, X, y):
        """
        feed the data to the NN object
        :param X: feature values. row represent a single data set
        :param y: expected value for a given data
        :return:
        """
        self.noOfDataSets = len(X)
        if self.noOfDataSets == 0:
            print("No data sets found. Terminating...")
            exit(-2)
        self.noOfFeatures = len(X[0])
        if self.typeOfNN == "Regression":
            for i in range(self.noOfDataSets):
                self.X = np.append(self.X, X[i])
                self.y = np.append(self.y, [y[i]])
        elif self.typeOfNN == "Classification":
                totalClasses = max(y)-min(y)+1
                for i in range(self.noOfDataSets):
                    self.X = np.append(self.X, X[i])
                    self.y = np.append(self.y, [int(y[i] == j) for j in range(totalClasses)])
        else:
            print("NN type mismatch")
            exit(-1)
        self.X = self.X.reshape(self.noOfDataSets, self.noOfFeatures)
        self.y = self.y.reshape(self.noOfDataSets, -1)

        if self.typeOfNN == "Regression":
            self.y, self.yMean, self.yMax, self.yMin = self.normalize(self.y)
        self.X, self.XMean, self.XMax, self.XMin = self.normalize(self.X)
        # self.X, self.W_white = self.whiten(self.X)

    def clearData(self):
        self.noOfDataSets = 0
        self.noOfFeatures = 0
        self.X = []
        self.y = []

    def normalize(self, mtx):
        mtx_mean = 0#mtx.mean(axis=0)
        mtx_max = mtx.max(axis=0)
        mtx_min = mtx.min(axis=0)
        mtx -= mtx_mean
        for i in range(len(mtx)):
            for j in range(len(mtx[0])):
                if (mtx_max[j] - mtx_min[j]) < 1e-5:
                    continue
                mtx[i][j] = (mtx[i][j] - mtx_min[j]) / (mtx_max[j] - mtx_min[j])
        return mtx, mtx_mean, mtx_max, mtx_min

    def whiten(self, X, fudge=1E-18):
        # get the covariance matrix
        Xcov = np.dot(X.T, X)

        # eigenvalue decomposition of the covariance matrix
        d, V = np.linalg.eigh(Xcov)

        # a fudge factor can be used so that eigen vectors associated with
        # small eigenvalues do not get over amplified.
        D = np.diag(1. / np.sqrt(d + fudge))

        # whitening matrix
        W = np.dot(np.dot(V, D), V.T)

        # multiply by the whitening matrix
        X_white = np.dot(X, W)

        return X_white, W

    def createLayer(self, ID, size, layerType='hidden', activationMethod="Sigmoid"):
        """
        creates a layer of given ID and size.
        :param activationMethod: Sigmoid / Rectifier
        :param ID: layer ID. usually initialize with integers. 0 - input, 1 .. n - hidden, n+1 - output
        :param size: layer size that you want to create. bias will be added automatically. do not consider it.
        :param layerType: type of the layer that is being created (input/hidden/output)
        :return:
        """
        self.layers[ID] = Layer(ID, size, activationMethod)
        if layerType == "hidden":
            self.hiddenLayerIDs.append(ID)
            self.layers[ID].node[0].activation = 1
        elif layerType == "input":
            self.inputLayerID = ID
            self.layers[self.inputLayerID].node[0].activation = 1
        elif layerType == "output":
            self.outputLayerID = ID
            self.layers[self.outputLayerID].node[0].activation = 1

    def linkLayer(self, layerID, connectFromLayerID, connectType="All"):
        """
        Link 2 layers by giving weight values to the nodes in a given layer
        :param layerID: layer that you want to connect
        :param connectFromLayerID: where does the links come from.
        :param connectType:
            "All" will connect all the nodes in the connectFrom layer to all the nodes in the current layer.
        :return:
        """
        self.layers[layerID].layerConnectedFrom.append(self.layers[connectFromLayerID])
        self.layers[connectFromLayerID].layerConnectedTo.append(self.layers[layerID])

        if connectType == "All":
            for nodeID in range(1, self.layers[layerID].layerSize):  # bias has no connections to come from.
                node2 = self.layers[layerID].node[nodeID]
                try:
                    node2.theta[connectFromLayerID].keys()
                except KeyError:
                    node2.theta[connectFromLayerID] = {}
                    node2.thetaHistory[connectFromLayerID] = {}
                for connectionFromNodeID in range(self.layers[connectFromLayerID].layerSize):
                    w = np.random.randint(1, 100)/100
                    node1 = self.layers[connectFromLayerID].node[connectionFromNodeID]
                    node2.theta[connectFromLayerID][connectionFromNodeID] = w
                    node2.thetaHistory[connectFromLayerID][connectionFromNodeID] = []
                    try:
                        node1.theta[layerID][nodeID] = w
                    except KeyError:
                        node1.theta[layerID] = {}
                        node1.theta[layerID][nodeID] = w
                        node1.thetaHistory[layerID] = {}
                        node1.thetaHistory[layerID][nodeID] = []
        elif connectType == "Convolution":
            pass

    def loss(self, Y):
        if len(Y) == 0:
            return 0

        weightSquareSum = 0
        for layerID in [self.inputLayerID] + self.hiddenLayerIDs:
            for connectedLayer in self.layers[layerID].layerConnectedTo:
                connectedLayerID = connectedLayer.layerID
                for startNodeID in range(self.layers[layerID].layerSize):
                    for endNodeID in range(1, self.layers[connectedLayerID].layerSize):
                        weightSquareSum += self.layers[layerID].node[startNodeID].theta[connectedLayerID][
                                               endNodeID] ** 2

        J = (self.regularizationFactor / 2) * weightSquareSum

        for i in range(len(self.currentEstimate)):
            J += self.cost(Y[i], self.currentEstimate[i])
        return J

    def cost(self, y_i, estimate_i):
        if self.cost_function == "quadratic":
            cost = 0
            for j in range(len(estimate_i)):
                cost += (y_i[j] - estimate_i[j]) ** 2
            return cost * 0.5

        elif self.cost_function == "cross-entropy":
            cost = 0
            for j in range(len(estimate_i)):
                if abs(estimate_i[j]) < 1e-320:
                    a = -1000
                else:
                    a = np.log(estimate_i[j])
                if abs(estimate_i[j] - 1) < 1e-320:
                    b = -1000
                else:
                    b = np.log(1 - estimate_i[j])
                cost -= y_i[j] * a + (1 - y_i[j]) * b

            return cost

        elif self.cost_function == "exponential":
            pass
        elif self.cost_function == "hellinger-distance":
            cost = 0
            for j in range(len(estimate_i)):
                cost += (y_i[j] ** 0.5 - estimate_i[j] ** 0.5) ** 2
            return cost * (0.5 ** 0.5)
        elif self.cost_function == "kullback-leibler-divergence":
            pass
        elif self.cost_function == "genaralized-kullback-leibler-divergence":
            pass
        elif self.cost_function == "itakura-saito-distance":
            pass

    def forward_prop(self, X, Y=None, back_prop=False):
        self.currentEstimate = []

        # reporting theta values
        for layer1ID in self.hiddenLayerIDs + [self.outputLayerID]:
            layer1 = self.layers[layer1ID]
            for connectionComingFromLayer in layer1.layerConnectedFrom:
                connectionComingFromLayerID = connectionComingFromLayer.layerID
                for nodeID in range(1, layer1.layerSize):  # skip bias node
                    for conNodeID in layer1.node[nodeID].theta[connectionComingFromLayerID].keys():
                        w = layer1.node[nodeID].theta[connectionComingFromLayerID][conNodeID]
                        layer1.node[nodeID].thetaHistory[connectionComingFromLayerID][conNodeID].append(w)

        for dataIndex in range(len(X)):

            # filling the input
            for inputLayerNodeID in range(1, self.layers[self.inputLayerID].layerSize):
                self.layers[self.inputLayerID].node[inputLayerNodeID].activation = X[dataIndex][inputLayerNodeID - 1]

            # forward propagating
            for hiddenLayerID in self.hiddenLayerIDs + [self.outputLayerID]:
                self.layers[hiddenLayerID].forwardToNextLayers()

            self.currentEstimate.append([])
            for outNodeID in range(1, self.layers[self.outputLayerID].layerSize):
                self.currentEstimate[-1].append(self.layers[self.outputLayerID].node[outNodeID].activation)

            if back_prop:
                self.back_prop(Y[dataIndex])

        self.currentEstimate = np.array(self.currentEstimate).reshape(len(X), -1)

        return self.currentEstimate

    def back_prop(self, y):
        # calculating delta for the output layer
        for outputNodeID in range(1, self.layers[self.outputLayerID].layerSize):
            activation = self.layers[self.outputLayerID].node[outputNodeID].activation
            activationPrime = self.layers[self.outputLayerID].activationFuncPrime(activation)
            if abs(activationPrime < 1e-50):
                activationPrime = 1e-5
            error = y[outputNodeID - 1] - activation

            if self.cost_function == "cross-entropy":
                self.layers[self.outputLayerID].node[outputNodeID].delta = error*-1#/(activation*(1-activation)) * -activationPrime
            elif self.cost_function == "quadratic":
                self.layers[self.outputLayerID].node[outputNodeID].delta = error * -activationPrime
            elif self.cost_function == "hellinger-distance":
                self.layers[self.outputLayerID].node[outputNodeID].delta = ((y[outputNodeID-1]**0.5 - activation**0.5)/(2*activation)**0.5) * -activationPrime

        IDs = [self.inputLayerID] + self.hiddenLayerIDs
        # calculating delta for the hidden layers
        for temp_i in range(len(IDs) - 1, -1, -1):
            startLayerID = IDs[temp_i]
            startLayer = self.layers[startLayerID]

            for endLayer in startLayer.layerConnectedTo:
                endLayerID = endLayer.layerID

                for startNodeID in range(startLayer.layerSize):
                    startNode = startLayer.node[startNodeID]

                    temp_delta = 0

                    for endNodeID in startNode.theta[endLayerID].keys():
                        endNode = endLayer.node[endNodeID]

                        theta = startNode.theta[endLayerID][endNodeID]
                        endNodeDelta = endNode.delta

                        temp_delta += theta * endNodeDelta

                    activation = startNode.activation
                    activationPrime = startLayer.activationFuncPrime(activation)

                    startNode.delta = temp_delta * activationPrime

        # adjusting theta values
        verySmallDelta = 0
        negativeDelta = 0
        for temp_i in range(len(IDs) - 1, -1, -1):
            startLayerID = IDs[temp_i]
            startLayer = self.layers[startLayerID]

            for endLayer in startLayer.layerConnectedTo:
                endLayerID = endLayer.layerID

                for startNodeID in range(startLayer.layerSize):
                    startNode = startLayer.node[startNodeID]
                    for endNodeID in startNode.theta[endLayerID].keys():

                        deltaOfConnectedNode = endLayer.node[endNodeID].delta

                        if abs(deltaOfConnectedNode) < 1e-5:
                            verySmallDelta += 1
                        elif deltaOfConnectedNode < 0:
                            negativeDelta += 1

                        activation = startNode.activation
                        theta = startNode.theta[endLayerID][endNodeID]

                        startNode.theta[endLayerID][endNodeID] -= self.learningRate * (
                            deltaOfConnectedNode * activation + self.regularizationFactor * theta)
                        endLayer.node[endNodeID].theta[startLayerID][startNodeID] -= self.learningRate * (
                            deltaOfConnectedNode * activation + self.regularizationFactor * theta)

        if negativeDelta:
            self.backproplog.write("Negative delta occurrences " + str(negativeDelta)+"\n")
        if verySmallDelta:
            self.backproplog.write("Very small delta occurrences " + str(verySmallDelta)+"\n")

    def learn(self, X, Y):
        self.forward_prop(X, Y, True)
