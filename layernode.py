import numpy as np


class Node(object):
    def __init__(self, layerID):
        """
        Node object to hold weights and other information.
        NOTE: nodeID are index number in a given layer
        :param layerID: layerID that the node belongs to.
        """
        self.layerID = layerID
        self.activation = 1
        self.delta = 0
        self.theta = {}
        self.thetaHistory = {}


class Layer(object):
    def __init__(self, ID, size, activationFunc="Sigmoid"):
        """
        :type activationFunc: the activation function of the layer
        :param ID: layer ID of the Layer object
        :param size: number of nodes in the Layer object excluding bias layer.
        """
        self.activationMethod = activationFunc
        self.layerID = ID
        self.layerSize = size + 1
        self.layerConnectedTo = []
        self.layerConnectedFrom = []
        self.node = {0: Node(ID)}  # bias node
        for i in range(size):
            self.node[i + 1] = Node(ID)

    def activationFunc(self, value):
        if self.activationMethod == "Sigmoid":
            return self.sigmoid(value)
        elif self.activationMethod == "Rectifier":
            return self.rectifiedLinear(value)
        elif self.activationMethod == "Linear":
            return self.linear(value)
        elif self.activationMethod == "Softplus":
            return self.softPlus(value)

    def activationFuncPrime(self, activation):
        if self.activationMethod == "Sigmoid":
            return activation * (1 - activation)
        elif self.activationMethod == "Rectifier":
            if activation <= 0:
                return 0
            else:
                return 1
        elif self.activationMethod == "Linear":
            return 1
        elif self.activationMethod == "Softplus":
            if activation == 0:
                return 0
            if activation > 100:
                return activation
            return self.sigmoid(np.log(np.exp(activation) - 1))

    @staticmethod
    def softPlus(z):
        if z>100:
            return z
        return np.log(1 + np.exp(z))
    @staticmethod
    def rectifiedLinear(z):
        return max(0, z)

    @staticmethod
    def linear(z):
        return z

    @staticmethod
    def sigmoid(z):
        if z < -700:
            return 0
        return 1 / (1 + np.exp(-z))

    def forwardToNextLayers(self):
        for connectionComingFromLayer in self.layerConnectedFrom:
            connectionComingFromLayerID = connectionComingFromLayer.layerID
            for nodeID in range(1, self.layerSize):  # skip bias node
                temp = 0
                for conNodeID in self.node[nodeID].theta[connectionComingFromLayerID].keys():

                    a = connectionComingFromLayer.node[conNodeID].activation
                    w = self.node[nodeID].theta[connectionComingFromLayerID][conNodeID]

                    temp += a * w
                self.node[nodeID].activation = self.activationFunc(temp)
