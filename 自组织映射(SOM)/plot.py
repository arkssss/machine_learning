import matplotlib.pyplot as plt

def plot_error():
    differ = []
    x = []
    times = 1
    with open("differ.txt") as f:
        for line in f.readlines():
            differ.append(float(line))
            x.append(times)
            times += 1

    plt.figure()
    plt.plot(x, differ)
    plt.title("Euclidean distance between its values in the t-th and (t+1)-th iterations")
    plt.savefig('error.svg')
    plt.show()

def plot_middle():
    neurons = []
    with open("middle.txt") as f:
        for line in f.readlines():
            each_number = line.strip().split(' ')
            image = []
            for number in each_number:
                image.append(float(number))
            neurons.append(image)

    times = 1
    image_x = 28
    for neuron in neurons:
        img = []
        for i in range(image_x):
            img.append(neuron[i*image_x:(i+1)*image_x])
        plt.subplot(10, 10, times)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        times += 1
    # if use show, the process will be blocked
    # plt.show()
    plt.savefig("middle_plot.svg")


def plot_converge():
    neurons = []
    with open("converge.txt") as f:
        for line in f.readlines():
            each_number = line.strip().split(' ')
            image = []
            for number in each_number:
                image.append(float(number))
            neurons.append(image)

    times = 1
    image_x = 28
    for neuron in neurons:
        img = []
        for i in range(image_x):
            img.append(neuron[i*image_x:(i+1)*image_x])
        plt.subplot(10, 10, times)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        times += 1
    # if use show, the process will be blocked
    # plt.show()
    plt.savefig("converge.svg")


if __name__ == '__main__':
    plot_middle()
    plot_converge()