

import pickle

from neural.trainer import *
from neural.nn import NN

'''
when data contain negative values, the nn fails.
if number of features affecting the output is less than what is given. cost reduces after lots of iterations.
numerical gradient-for theta(i,j) + eps you need to forward prop and get the estimate!
softplus goes to inf -- DONE!
pause in the middle
convolution

polyak averaging
momentum
'''


def createData(equation, variables, size, scatter):
    x_data = []
    y_data = []
    for i in range(size):
        temp = equation
        x_data.append([])
        for j in range(len(variables)):
            x_data[-1].append(np.random.randint(1, 200)/100)
            temp = temp.replace(variables[j], str(x_data[-1][-1]))
        y_data.append(eval(temp))
    x_data = np.array(x_data)
    y_data = 1.*np.array(y_data)
    y_data += np.random.normal(size=y_data.shape, scale=((sum(y_data*y_data)**0.5)/size)*scatter/100)
    return x_data, y_data


def createDataClassification(features, classes, size, scatter):
    '''
    note that the range of a feature is 0 to 1
    :param features: number of features to explain a point
    :param classes: number of classes in the dataset
    :param size: size of the data set
    :param scatter: scatter parameter
    :return:
    '''
    x_data = []
    y_data = []
    centers = [[] for _ in range(classes)]
    for i in range(classes):
        for j in range(features):
            centers[i].append(np.random.randint(0,1000)/1000)
    for i in range(size):
        x_data.append([])
        c = np.random.randint(0,classes)
        y_data.append(c)
        for j in range(features):
            x_data[-1].append(centers[c][j] + np.random.normal()*scatter)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def exportNNdata(NN):
    fh = open("nn_data.txt", "w")
    for layerID in [NN.inputLayerID] + NN.hiddenLayerIDs:
        for connectedLayer in NN.layers[layerID].layerConnectedTo:
            connectedLayerID = connectedLayer.layerID
            fh.write("layer {} connected to {}\n".format(layerID, connectedLayerID))
            for startNodeID in range(NN.layers[layerID].layerSize):
                for endNodeID in range(1, NN.layers[connectedLayerID].layerSize):
                    fh.write("{} {} -> {}\n".format(startNodeID,NN.layers[layerID].node[startNodeID].theta[connectedLayerID][endNodeID], endNodeID))


def createNN(typeOfNN, costf, inputsize, outputsize, hiddensize, hiddenlayers, hiddenactivation):
    neuralNet = NN(typeOfNN,costf)
    neuralNet.createLayer(0, inputsize, "input")
    for i in range(hiddenlayers):
        neuralNet.createLayer(i+1, hiddensize, activationMethod=hiddenactivation)
    neuralNet.createLayer(hiddenlayers+1, outputsize, "output", "Sigmoid")
    for i in range(hiddenlayers+1):
        neuralNet.linkLayer(i+1,i)
    return neuralNet

if __name__ == "__main__":
    nnCreateTime = time.time()

    neuralNet = createNN(typeOfNN="Classification", costf="cross-entropy",inputsize=2, outputsize=6, hiddensize=6, hiddenlayers=3, hiddenactivation="Sigmoid")

    print("Neural network created in {} secs.".format(time.time() - nnCreateTime))

    data = unpickle("testdata/data_batch_1")

    """inputX = data[b'data'][:10]
    inputY = data[b'labels'][:10]
    for i in range(len(inputY)):
        inputY[i] = [int(j==inputY[i]) for j in range(10)]"""
    inputX, inputY = createData('np.sin(y)**2 + np.exp(z) + z*y + z**0.5 + y*8 + 2', ['z', 'y'], 100, 10)
    inputX, inputY = createDataClassification(features=2, classes=6, size=100, scatter=0.1)
    print("Feeding data into the Neural Network...")
    neuralNet.feed(inputX, inputY)
    print("Data feeding complete.")

    T = Trainer(neuralNet, 0.1, 0.00001, view=False, verify=False)

    #T.analyzeLearningRate()

    trainTime = time.time()
    print("Neural network training....")
    #T.train(30,100,2)
    T.trainWithoutTesting(200)
    print("Training completed in {} secs".format(time.time() - trainTime))




    exportNNdata(T.NN)

    T.plotCost()
    T.plotThetaHistory()

    T.contourAnd3dPlot()
    plt.show()
