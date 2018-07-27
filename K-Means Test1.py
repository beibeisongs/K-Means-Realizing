# encoding=utf-8
# Date: 2018-7-22
# Author: MJUZY


"""
The Analysis of K-Means algorithm:
    * Advantages:
        Easy to be Realized
    * Disadvantages:
        Probable to Converge to a local minimum value

    Converge to a local minimum value: 收敛到局部最小值

    * IDEA of K-Means:
        First, we should choose a value K
            The K refers to the num of the Clusters we will create later
        Second, we should choose the original Centroid

        Centroid: n 质心 重心 聚类点

            The Centroid is generally chosen at Random
            And the Centroid here is chosen among the Data Region at Random
                And the another method is to choose one of the Sample of the DataSet
                    But it may cause Converging to a local minimum value
                        So we do not use the second Method

    * The Notes of K-Means Realising by myself:
        Then we will compute every Distance between each Sample and the Centroid
            and Classify the Samples to the Centroid when they have the Minimun Distance
        After Completing the Step above, we will repeat the same step
            util the result stops to Converge

    * Recording the Functions:

        def loadDataSet(filename)
            # Read the DataSet from the file
        def distEclud(VecA, VecB)
            # Compute the Distances
                # Using Euclidean Distance
                    # In Addition, Other kinds of reasonable Distance are OK, too
        def randCent(dataSet, k)
            # Create Centroids at Random
                # Choose a specific Point in the Data Region
        def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent)
            # K-Means Algorithm: Input the DataSet and the K value
                # The next two Parameters respectively are the Computing Method optional and the Method to choose Centroids
        def show(dataSet, k, centroid, clusterAssment)
            # Make the Result visualized
"""


from numpy import *
import csv


def loadDataSet(filename):
    """
        :param label: an One Dimension Array
        :param dataSet: A Multi-Dimension Matrix
        :return: dataSet, label
        """
    dataSet = []
    with open(filename, "r") as csvfile:
        reader2 = csv.reader(csvfile)  # 读取csv文件，返回的是迭代类型
        for item2 in reader2:
            dataSet.append(item2)
    csvfile.close()

    label = ['特征一', '特征二', '特征三', "特征四"]

    return dataSet


def randCent(dataSet, k):
    """
    Create Centroids at Random

    :param dataSet:
    :param k: refers to the num of Centroids to be created
    :return:
    """
    # <Sample>: dataSet.shape = <class 'tuple'>: (150, 4) n = 4
    # Get the num of the Features of the Samples
    n = shape(dataSet)[1]

    # <Description>: create a k * n matrix to store the k centroids of n dimensions
    centroids = mat(zeros((k, n)))

    for j in range(n):
        # ---------------------------------------------- There is a Wonderful Coding Technique !
        minJ = min(dataSet[:, j])
        rangeJ = float(max(array(dataSet)[:, j]) - minJ)

        # ----------------------------------------------
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1) # Get Numbers from [0, 1]
                                                            # And rand(k, 1) means the Array Size of k * 1
    return centroids


def distEclud(vecA, vecB):
    """
    Compute the Euclidean distance of the Vectories

    :param vecA:
    :param vecB:
    :return:
    """
    power_value = power(vecA - vecB, 2)

    sum_value = sum(power_value)

    result = sqrt(sum_value)

    return result


def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):

    # <Sample>: dataSet.shape = <class 'tuple'>: (150, 4) m = 150
    m = shape(dataSet)[0]

    """
    Functions of "zeos()":
        Create an Array whose elements are all 0
        
        Samples:
            >>myMat=np.zeros(3)    # Create an Array with elements all 0
            >>print(myMat)
            >>array([0.,0.,0.])
            
            >>myMat1=np.zeros((3,2))  # Create an 3 * 2 Array with elements all 0
            >>print(myMat)
            >>array([[0.,0.],
                    [0.,0.]
                    [0.,0.]])
    
    clusterAssment = mat(zeros((m, 2)))
    
        Used to create matrix to assign data points
            to a centroid, also holds SE of each point
    """
    clusterAssment = mat(zeros((m, 2))) # <Description>: So the size of the Matrix
                                        #   clusterAssment is m * 2
                                        #   During this Sample

    centroids = createCent(dataSet, k)

    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  #  for each data Point assign it to the Closest Centroid
            minDist = inf   # inf = Infinity n 无穷大
            minIndex = -1
            for j in range(k):
                """
                centroids[j, :] get all elements of the specific dimension
                dataSet[i, :] get all elements of the specific dimension
                
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                    Get the Distance of the two Vectories
                """
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI    # <Description>: minDist/distJI is the Euclidean distance of the two Vectories

                    minIndex = j    # <Attention>:
                                    #   Now the j's range is [0, k]

                    # The most ideal situation is that all the clusterAssment[i, 0] = minIndex

            # When all the clusterAssment[i, 0] equals minIndex
            #   The Loop will be stopped
            if clusterAssment[i, 0] != minIndex:
                """<Attention>:
                    So the size of the Matrix
                        clusterAssment is m * 2
                            During this Sample, its size is 150 * 2
                """
                clusterChanged = True   # So the "while" Loop will be continued

            # <Description>: Symbol "** 2" means to Calculate the Square
            # <Attenion>;
            #   the Size of the list "clusterAssment" is m * 2
            #   During this Sample m * 2 equals 150 * 2
            #   [i, :] = minIndex, minDist **2, after that, it has only 2 elements in each Dimension
            clusterAssment[i, :] = minIndex, minDist **2

            print("Centroid Temporary : ", centroids)

        for cent in range(k):   # Recalculate the Centroids
                                # <Attention>: k is the num of the clusters will be created

            """
            Get all the Points in this cluster
            
            if you want to Grasp the logic Details clearly 
                please debug the codes following:
                
            """
            Get_FirstElement_intoArray = clusterAssment[:, 0].A
            Get_Matched_Array = (Get_FirstElement_intoArray == cent)
            operate_nonzero = nonzero(Get_Matched_Array)
            Get_FirstElement_Again = operate_nonzero[0]
            ptsInClust = dataSet[Get_FirstElement_Again]

            # <Description>: correct the Temporary Centroids by the Average Distance Points
            centroids[cent, :] = mean(ptsInClust, axis=0)

    return centroids, clusterAssment


def show(dataMat, k, centroids, clusterAssment):
    from matplotlib import pyplot as plt

    numSamples, dim = dataMat.shape

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']

    """
    Attention:
        PLT can only show the Clusters in a TWO-DIMNSION Graph
    """
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataMat[i, 0], dataMat[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


def main(k):
    """
    :param k: whick is set for the K-Means before hand
    :param dataMat: type: matrix
    :return NULL
    """
    # Funtion mat() can transform the paramter's data type from list to matrix
        # Specifically refering: http://www.cnblogs.com/MrLJC/p/4127553.html
    # <Attention>:
    #   Function "mat()" will make the elements of the Matrix string
    #   Function "astype(DataType)"will make the elements of the Matrix DataType
    dataMat = mat(loadDataSet('iris_Test1.csv')).astype(float)
    print("This is the  original Data Matrix : ", dataMat)

    centroids, clusterAssment = kMeans(dataMat, k)
    print("My Centroids are : ", centroids)

    show(dataMat, k, centroids, clusterAssment)

if __name__ == "__main__":
    """
    :param k: this is the num of the clusters you want to create
    """
    k = 3

    main(k)