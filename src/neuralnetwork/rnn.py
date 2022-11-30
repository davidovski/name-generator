import random
import time
import math
import torch
import torch.nn as nn

import warnings
import sys

import copy

from util import AtomicNumber

PRINT_INFORMATION_SECONDS = 2
num_processes = 12

#ignore warnings
warnings.filterwarnings('ignore')


if "--disable-cuda" in sys.argv:
    cuda = False
else:
    cuda = torch.cuda.is_available()

print(f"CUDA is {'enabled' if cuda else 'disabled'}")
if cuda:
    print("CUDA devices:")
    for device_index in range(torch.cuda.device_count()):
        print(f"{device_index}|\t{torch.cuda.get_device_name(device_index)}")

device = torch.device("cuda") if cuda else torch.device("cpu")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # create the input, hidden and output linear transformation branches
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size, device=device)
        self.input_to_output = nn.Linear(input_size + hidden_size, output_size, device=device)
        self.output_to_output = nn.Linear(hidden_size + output_size, output_size, device=device)

        # initialise a dropout function to be used on output data
        self.dropout = nn.Dropout(0.1)

        # initialise the softmax function to be used on output data
        self.softmax = nn.LogSoftmax(dim=1)
            
        # do not introduce any randomness by default
        self.random_factor = 0


    def forward(self, inputs, hidden):
        # combine the input layer with the hidden layer to create the output layer and new hidden layer
        input_combined = torch.cat((inputs, hidden), 1)
        hidden = self.input_to_hidden(input_combined)
        output = self.input_to_output(input_combined)
        output_combined = torch.cat((hidden, output), 1)

        output = self.output_to_output(output_combined)
        # apply the functions to the output data
        output = self.dropout(output)
        output = self.softmax(output)
        
        # add noise to the output, based on self.random_factor
        if self.random_factor > 0:
            # create a fully random tensor
            random_tensor = torch.randn(self.output_size)
            output = torch.add(output, random_tensor, alpha=self.random_factor)

        return output, hidden

    def initHidden(self):
        # The hidden layer should be tensor with the length that we've specified
        return torch.zeros(1, self.hidden_size, device=device)

# instantiate the function to use to calculate loss
#   we will use Mean Squared Error between the 
criterion = nn.NLLLoss()

# define the learning rate, to begin with, we can use 0.0005
learning_rate = 0.0005

"""Train a neural network on a single input name
    Args:
        rnn: (RNN) the rnn to train
        input_tensors: (tensor) The input tensor: a one-hot-encoding from the first letter to the last letter, excluding the end of string marker
        output_tensors: (tensor) The input tensor: a one-hot-encoding from the second letter to the end of the input data
    Returns:
        output: (tensor) the output of the training
        loss:   (float)  the loss of the training
"""
def train_rnn(rnn, input_tensor, target_tensor):
    # unsqueeze the target tensor, 
    target_tensor.unsqueeze_(-1)
    
    # reset the parameters of the neural network
    hidden = rnn.initHidden()
    rnn.zero_grad()

    # initiate an float called loss, this will store the error between each iteration output and its target
    loss = 0
    for i in range(input_tensor.size(0)):
        output, hidden = rnn(input_tensor[i], hidden)

        # calculate the error and add it to the overall loss
        l = criterion(output, target_tensor[i])
        loss += l

    loss.backward()

    # adjust the parameters of the rnn accordingly
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_tensor.size(0)

"""Create the input tensor for a name, a one hot matrix from the first letter to last letter (excluding EOS)
    Args:
        name: (str[]) an array of the letters in the name, can also be supplied as a string literal
        alphabet: (str[]) The alphabet to use while encoding the name, an array starting with a "NULL" character and ending in an "EOS" character
        value: (float) (default=1) The value to use for the "1" representing the letter 
    Returns:
        tensor: (tensor) the input tensor for the given name
"""
def input_tensor(name, alphabet, value=1):
    tensor = torch.zeros(len(name), 1, len(alphabet), device=device)

    #iterate through each letter in the name
    for li in range(len(name)):
        letter = name[li]
        # If the letter isn't in the alphabet, use the first "NULL" character
        index = alphabet.index(letter) if letter in alphabet else 0 

        tensor[li][0][index] = value
    
    return tensor

"""Create the target tensor for a name, a long tensor from the second letter to the EOS
    Args:
        name: (str[]) an array of the letters in the name, can also be supplied as a string literal
        alphabet: (str[]) The alphabet to use while encoding the name, an array starting with a "NULL" character and ending in an "EOS" character
    Returns:
        tensor: (tensor) the input tensor for the given name
"""
def target_tensor(name, alphabet):
    indexes = []
    for li in range(1, len(name)):
        letter = name[li]
        index = alphabet.index(letter) if letter in alphabet else 0 
        indexes.append(index)

    # and add the end of string character
    indexes.append(len(alphabet) - 1) 

    #legacy tensor needs to be made this way
    if cuda:
        return torch.cuda.LongTensor(indexes)
    else:
        return torch.LongTensor(indexes)


"""Train a neural network on a list of names with a given alphabet
    Args: 
        rnn (RNN): the neural network to train on
        names: (str[]) the list of names to train on
        alphabet: (str[]) the alphabet to use to encode characters
        iterations: (int) (default=10000) The number of iterations of training that should be done
"""
def learn_names(rnn, names, alphabet, iterations=100000, num_processes=12):
    
    # keep track of total time spent training by knowing when we started training
    start = time.time()

    # define the number of iterations per process
    iters_per_process = int(iterations/num_processes) 

    processes = []

    # keep track of the total loss
    total_loss = AtomicNumber()

    # keep track of total number of completed iterations
    completed_iterations = AtomicNumber()

    # keep track of the last time that the information was printed
    #   this way we can print every x seconds
    last_print = AtomicNumber()

    print(f"Training on {len(names)} names...")

    # spawn processes, each running the _train function
    torch.multiprocessing.spawn(_train, args=(rnn, names, alphabet, iters_per_process,
            total_loss, completed_iterations, last_print, start, iterations),
            nprocs=num_processes,
            join=True)
    print()

"""Thread function to use when multiprocessing learn_names

"""
def _train(rank, rnn, names, alphabet, iterations, 
            total_loss, completed_iterations, last_print,
            start, total_iterations):
    for i in range(1, iterations+1):
        try:
            # choose a random name to train on
            name = random.choice(names) 

            # create the input and trainint tensors
            input_name_tensor = input_tensor(name, alphabet)
            target_name_tensor = target_tensor(name, alphabet)

            # train the rnn on the input and target tensors 
            output, loss = train_rnn(rnn, input_name_tensor, target_name_tensor)
            total_loss.increment(loss)

            # increment number of completed iterations
            completed_iterations.increment()

            # to prevent overloading the console, potentially slowing down the training process,
            #   only print information every PRINT_INFORMATION_SECONDS
            if time.time() - last_print.get() > PRINT_INFORMATION_SECONDS:
                # set last print to now to prevent other threads from also printing
                last_print.set(time.time())

                # calculate and display information
                seconds_elapsed = time.time() - start
                time_elapsed = "%dm %ds" % (math.floor(seconds_elapsed / 60), seconds_elapsed % 60)

                percentage = completed_iterations.get() / total_iterations * 100

                # print information on the same line as before
                print("\r%s (%d %d%%) %.4f" % (time_elapsed, completed_iterations.get(), percentage, loss), end="")
        except:
            pass


"""Sample a random name from the network using a starting letter
    Args:
        rnn: (RNN) the neural network to sample from
        alphabet: (str[]) the alphabet to use to decode the outputs from the network
        start_letter: (str) the letter to use to start the neural network
        max_length: (int) (default=50) the maximum length for a name
    Returns:
        output_name: (str) the characters that the rnn has generated from the starting letter
"""
def sample(rnn, alphabet, start_letter, max_length=50):
    # disable gradient calculation
    #with torch.no_grad():
        # create the input tensor from the start letter, using a randomized value
        #random_value = random.random()
        random_value = 1 
        sample_input = input_tensor(start_letter, alphabet, value=random_value)
    
        rnn.dropout(sample_input)

        # reset hidden layer
        hidden = rnn.initHidden()

        output_name = [start_letter]
    
        # use a max length to prevent names from being too long
        for i in range(max_length): 
            # call the rnn for the next letter
            output, hidden = rnn(sample_input[0], hidden)
            
            top_v, top_i = output.topk(1)
            top_i = top_i[0][0]

            if top_i == len(alphabet)-1: # EOS has been reached
                break;
            else: 
                # append next letter to output

                letter = alphabet[top_i]
                output_name.append(letter)
                
                sample_input = input_tensor(letter, alphabet)

        return output_name


import warnings
# testing
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        english_alphabet = [c for c in " abcdefghijklmnopqrstuvwxyz"]
        english_alphabet.append("") # add the EOS character


        option = input("(t)rain or (s)ample?")
        if option == "t":

            names = []
            with open("data/datasets/usa/surname.txt", "r") as datafile:
                # convert all names to lowercase and remove newline character
                names = [name[:-1].lower() for name in datafile.readlines()]

            # create the neural network with a hidden layer of size 128
            rnn = RNN(len(english_alphabet), 128, len(english_alphabet))
            
            # transfer to cuda if cuda is enabled
            if cuda:
                rnn.cuda()

            def provide_name():
                return random.choice(names)

            learn_names(rnn, names, english_alphabet, iterations=100000, num_processes=12)
            print()


            torch.save(rnn, "data/english_names.pt")
        elif option == "s":
            rnn = torch.load("data/english_names.pt")
            if cuda:
                rnn.cuda()
            rnn.eval()
            rnn.random_factor = 0.7

            for start_letter in [i for i in "abcdefghijklmnopqrstuvwxyz"]:
                print(sample(rnn, english_alphabet, start_letter))
        else:
            print("invalid option!")
