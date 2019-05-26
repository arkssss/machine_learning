# Author : zhou fang
# Course : CSCI964 
# Email  : 522500442@qq.com
# Date   : 01/05/2019
import random
import math
import matplotlib.pyplot as plt
import copy

# set number of 2d lattice
rowNum = 10
colNum = 10

# the number of training neuron
train_class = rowNum * colNum

# neuron list
neurons = []
old_neurons = []

#
image_x = 28
image_y = 28

# training times
epochNumOrd = 100
epochNumCov = 100
train_times = epochNumOrd + epochNumCov

# training parameters
# Initialise the sigma value of the neighbourhood function h();
# sigma_0 is initialised as the "radius" of the lattice;
sigma_0 = math.sqrt(pow(0.3*rowNum, 2) + pow(0.3*colNum, 2))
sigma_t = sigma_0

# change the tau_1
tau_1 = 1000/math.log(sigma_0)

# Initialise the learning rate and define the rate in each epoch t;
eta_0 = 0.6
eta_t = eta_0
# Define the attenuation speed of the learning rate with epoches;
tau_2 = 1000


def read_mnist():
    """
    read the data set and return a list
    Each example column of the file (not row)
    row => 784  (features)
    col => 5000 (example)
    :return:
    examples : the number of train data
    dim      : the dim of one data (features)
    """
    res = []
    with open('SOM_MNIST_data.txt', 'r') as f:
        for line in f.readlines():
            res.append(line.split(' '))
    examples = len(res[0])
    dim = len(res)
    return res, examples, dim


def caldis(vector_one, vector_two):
    """
    cal the Euclidean distance between two vector
    :param vector_one:
    :param vector_two:
    :return: distance
    """
    len1 = len(vector_one)
    len2 = len(vector_two)
    if len1 != len2:
        raise RuntimeError("need same dim with the vector")
    dis = 0
    for i in range(len1):
        dis += pow((vector_one[i] - vector_two[i]), 2)
    return math.sqrt(dis)


def getWinner(data):
    """
    find the winner neuron in the neurons
    :param data:
    :return:
    """
    min_value = 100000  # save the min distance
    min_index = -1  # save the min neuron index
    for i in range(len(neurons)):
        dis = caldis(neurons[i], data)
        if dis < min_value:
            min_value = dis
            min_index = i
    return min_index


def get_2d_vector(neuron_index):
    """
    pass the index of the neuron
    return the 2d coordinates form
    [
    [0, 1, 2, ..., 9]
    [10,11,12 ... ,19]
    ...
    ]
    e.g.
    2 => [1, 3]
    11 => [1, 2]
    :param neuron_index:
    :return:
    """
    rowIndex = int(neuron_index / rowNum) + 1
    colIndex = (neuron_index % colNum) + 1
    return [rowIndex, colIndex]


def plot_one(vector, save_name):
    """
    vector => 28 * 28
    :param vector:
    :return:
    """
    img = []
    for i in range(image_x):
        img.append(vector[i * image_x:(i + 1) * image_x])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(save_name)


def plot_neuron(title="", save_name="test.svg"):
    """
    plot all the neuron
    :return:
    """
    times = 1
    for neuron in neurons:
        img = []
        for i in range(image_x):
            img.append(neuron[i*image_x:(i+1)*image_x])
        plt.subplot(10, 10, times)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        times += 1
    plt.suptitle(title)
    # if use show, the process will be blocked
    # plt.show()
    plt.savefig(save_name)


def save_diff(differ):
    """
    record the differ by save into a file
    :param differ: 
    :return: 
    """
    file_name = "differ.txt"
    with open(file_name, 'a') as f:
        f.writelines(str(differ) + "\n")


if __name__ == '__main__':
    DataSet, examples, data_dim = read_mnist()
    # reshape
    # images is DataSet do the Transpose
    images = []
    for i in range(0, examples):
        image = []
        for j in range(0, data_dim):
            image.append(float(DataSet[j][i]))
        images.append(image)

    # init the weight of each neuron to (0, 1)
    random.seed()
    for j in range(0, train_class):
        class_vector = []
        for i in range(0, data_dim):
            class_vector.append(round(random.random(), 5))
        neurons.append(class_vector)
    # old_neurons init
    old_neurons = copy.deepcopy(neurons)
    # plot init weight
    plot_neuron("init weight", "init_weight.svg")
    # start training
    for each_epoch in range(0, train_times):
        print("now you are at epoch : " + str(each_epoch))
        # randomly order at beginning of each epoch
        for i in range(examples):
            random_num = random.randint(0, examples-1)
            images[i], images[random_num] = images[random_num], images[i]
        times = 0
        for each_image in images:
            # get the winner
            times += 1
            winner_index = getWinner(each_image)
            # get the distance of the winner_index to all the neuron
            # update the weight
            for i in range(len(neurons)):
                # only the nearly neuron will be updated
                dis = caldis(get_2d_vector(i), get_2d_vector(winner_index))
                if dis < sigma_t:
                    for j in range(len(neurons[i])):
                        effect = math.exp(-dis / (2 * sigma_t * sigma_t))
                        neurons[i][j] += eta_t * effect * (each_image[j] - neurons[i][j])

        # differ the weight
        dis = 0
        for i in range(len(neurons)):
            dis += caldis(neurons[i], old_neurons[i])
        save_diff(dis)
        old_neurons = copy.deepcopy(neurons)

        # after each epoch update the parameters
        # Attenuate the learning rate for the next epoch
        # Note that the learning rate shall remain above 0.01
        eta_t = max(eta_0 * math.exp(-(each_epoch+1)/tau_2), 0.01)

        # Attenuate the sigma value of h() for the next epoch;
        sigma_t = sigma_0 * math.exp(-(each_epoch+1)/tau_1)

        # plot
        plot_neuron("after epoch : " + str(each_epoch + 1), "epoch_"+str(each_epoch+1)+".svg")