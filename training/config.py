# Hyperparameters --------------------
input_size = 784 # no. of units in image (28x28=784)
num_classes = 10 # no. of classes (i.e model will output any 1 integer out of 0, 1, 2, ....., 9)
learning_rate = 0.001 # learning rate alpha
batch_size = 64 # mini-batch size
num_epochs = 1 # no. of epochs

# Dataset
data_dir = 'dataset/'
num_workers = 4

# Compute related
accelerator = "gpu"
devices = 1
precision = 16
