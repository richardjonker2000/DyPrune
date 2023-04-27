import json
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torchmetrics
from matplotlib.patches import BoxStyle
from matplotlib.patches import FancyBboxPatch
from matplotlib.pyplot import figure
from torch import nn
from torch.nn.utils import prune
from tqdm.auto import tqdm



def graph_visualizer(number_of_layer, neurons, weight_pruning):
    """
    Function to visualize a neural network
    :param number_of_layer: number of layers in the network, including input and output
    :param neurons: an array of size number_of_layers that contains the number of neurons in each layer
    :param weight_pruning: the weights that are pruned from the model
    :return:
    """

    if len(neurons) != number_of_layer:
        raise Exception("Each layer needs to have the number of neurons defined")

    # color palettes for the graph defined
    palette_pastel = ["#fddaec", "#fbb4ae", "#fed9a6", "#ffffcc", "#ccebc5", "#bde2e5", "#b3cde3", "#decbe4"]
    palette_normal = ["#d40086", "#d63800", "#fbb034", "#ffdd00", "#c1d82f", "#00cedd", "#005ae4", "#9a00e4"]

    # initialize the size of the image
    figure(figsize=(2 * number_of_layer, 8), dpi=80)

    # create an array of limited neurons for large nn -> at most 19 neurons
    neurons_reduced = []
    for i in neurons:
        if i < 20:
            neurons_reduced.append(i)
        else:
            neurons_reduced.append(3)

    # Create DAG
    G = nx.DiGraph()

    # Node list - a node for each neuron
    node_list = ['v' + str(i) for i in range(sum(neurons_reduced))]
    # Add nodes
    G.add_nodes_from(node_list)
    # colors for nodes -> each layer has a different color
    node_colors = []
    for i in range(number_of_layer):
        node_colors.extend([palette_normal[i % len(palette_normal)] for f in range(neurons_reduced[i])])

    # edges are for each node between each layer
    edge_list = []
    current_neuron = 0

    for layer in range(number_of_layer - 1):
        if neurons_reduced[layer] == neurons[layer]:
            for i in range(neurons_reduced[layer]):
                for j in range(neurons_reduced[layer + 1]):
                    if weight_pruning[layer][j][i].item() != 0:
                        edge_list.append((f'v{current_neuron + i}', f'v{current_neuron + j + neurons_reduced[layer]}'))

        else:
            for i in range(neurons_reduced[layer]):
                for j in range(neurons_reduced[layer + 1]):
                    edge_list.append((f'v{current_neuron + i}', f'v{current_neuron + j + neurons_reduced[layer]}'))
        current_neuron += neurons_reduced[layer]
    # add edges to graph
    G.add_edges_from(edge_list)
    # print(edge_list)

    # create the positions of the neurons
    pos = {}
    # start x so that 0 is centered
    start_x = - number_of_layer // 2
    # counter for the current neuron
    current_neuron = 0
    # relative padding
    padding = max(neurons_reduced) * 0.04

    for i in range(number_of_layer):
        if neurons_reduced[i] != neurons[i]:
            # if we have a reduced layer put the 2 neurons on the max y range, and one on 0
            pos[f'v{current_neuron}'] = (start_x, max(neurons_reduced) // 2)
            pos[f'v{current_neuron + 1}'] = (start_x, 0)
            pos[f'v{current_neuron + 2}'] = (start_x, -(max(neurons_reduced) // 2))
            current_neuron += 3
            # plot the 3 black dots between the neurons
            temp_y = np.linspace((max(neurons_reduced) // 2 - padding * 4.5), padding * 4.5, 3)
            plt.plot([start_x for i in range(3)], temp_y, 'o', color='black', markersize=3)
            temp_y = np.linspace((- padding * 4.5), (-(max(neurons_reduced) // 2) + padding * 4.5), 3)
            plt.plot([start_x for i in range(3)], temp_y, 'o', color='black', markersize=3)

            # color the background
            plt.gca().add_patch(FancyBboxPatch((start_x - 0.1, -(max(neurons_reduced) // 2) - padding), width=0.2,
                                               height=(max(neurons_reduced) // 2 + padding) * 2,
                                               boxstyle=BoxStyle("Round", pad=0.1),
                                               facecolor=palette_pastel[i % len(palette_pastel)], edgecolor="None",
                                               zorder=0))

        else:
            # define the start y based on how many items need to be processed
            start_y = ((neurons_reduced[i]) // 2) * (max(neurons_reduced) / neurons_reduced[i])
            # if there is an even amount  of neurons adapt the start by lowering it
            if neurons_reduced[i] % 2 == 0:
                start_y -= (max(neurons_reduced) / neurons_reduced[i]) / 2
            # process the neurons
            for j in range(neurons_reduced[i]):
                pos[f'v{current_neuron}'] = (start_x, start_y)
                current_neuron += 1
                start_y -= max(neurons_reduced) / neurons_reduced[i]
            # color the background
            plt.gca().add_patch(FancyBboxPatch((start_x - 0.1, -(max(neurons_reduced) // 2) - padding), width=0.2,
                                               height=(max(neurons_reduced) // 2 + padding) * 2,
                                               boxstyle=BoxStyle("Round", pad=0.1),
                                               facecolor=palette_pastel[i % len(palette_pastel)], edgecolor="None",
                                               zorder=0))

        # label the layers
        if i == 0:
            plt.text(start_x - 0.15, -(max(neurons_reduced) // 2) - 2 * padding, f"In: {neurons[i]}")
        elif i == number_of_layer - 1:
            plt.text(start_x - 0.17, -(max(neurons_reduced) // 2) - 2 * padding, f"Out: {neurons[i]}")
        else:
            plt.text(start_x - 0.17, -(max(neurons_reduced) // 2) - 2 * padding, f"HL-{i}: {neurons[i]}")

        start_x += 1

    # draw the graph
    nx.draw(G, pos=pos, with_labels=False, node_size=400, node_color=node_colors, arrows=False)

    # outline the nodes
    plt.gca().collections[0].set_edgecolor("#444444")
    plt.gca().collections[0].set_linewidth(0.35)

    # plt.title('DNN for iris') #Picture title - todo
    # save plot -todo

    plt.show()


class DynamicNeuralNetwork(nn.Module):
    def __init__(self, number_of_layers, neurons, loss=nn.CrossEntropyLoss(),
                 optimizer=torch.optim.AdamW, learning_rate=1e-3,
                 pruning="dynamic", pruning_iter=4, pruning_rate="dynamic", pruning_type="l1", pruning_min=0,
                 pruning_max=0.05,
                 mode="script", plot_model=False):
        """
        initialises the neural network, without the input layer. This is determined during runtime with pytorch
        LazyLinear.

        :param number_of_layers: how many layers the neural network has, excluding the input layer
        :param neurons: an array containing the number of output neurons each layer has
        :param loss: loss function

        :param optimizer: optimizer
        :param learning_rate: learning rate for the optimizer

        :param pruning: The type of purning to be executed. Must be one of ["dynamic", "static", False].
        :param pruning_iter: If pruning is set to static, the pruning iteration must be set.
        This is after how many epochs pruning will occur
        :param pruning_rate: The rate of pruning, must be either "dynamic" or a float between 0 and 1 reprsenting the
        pruning rate
         that wil be executed on the model.
        :param pruning_type: the method of which pruning is applied on the model must be one of   ["l1", "random"].
        "l1" corresponds to the l1unstructured method which uses the l1 scores to prune parameters (parameters closest
        to 0).
        "random" randomly selecs weights to prune
        :param pruning_min: The minimum relative gradient for which pruning will occur when using dynamic pruning
        :param pruning_max: The maximum relative gradient for which pruning will occur when using dynamic pruning
        """
        print(f"CUDA available:{torch.cuda.is_available()}")
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Number of CUDA devices:{torch.cuda.device_count()}")
            print(f"Current CUDA device number:{torch.cuda.current_device()}")
            print(f"Current CUDA device name:{torch.cuda.get_device_name(torch.cuda.current_device())}")

        # TODO final variable checks
        pruning_commands = ["dynamic", "static", False]
        pruning_type_commands = ["l1", "random"]
        growing_commands = ["dynamic", "static", False]
        neuron_init_param = ["xavier_uniform", "xavier_normal", "ones", "zeros", "average"]

        self.pruning_iter = pruning_iter
        self.pruning_rate = pruning_rate
        self.pruning_min = pruning_min
        self.pruning_max = pruning_max

        self.prune_time = 0
        self.add_neuron_time = 0
        self.add_layer_time = 0

        if pruning not in pruning_commands:
            raise Exception(f"Pruning must be one of {pruning_commands}")
        if pruning_type not in pruning_type_commands:
            raise Exception(f"Pruning_type must be one of {pruning_type_commands}")

        self.number_of_layers = number_of_layers + 1
        self.neurons_per_layer = [1]
        self.neurons_per_layer.extend(neurons)
        self.pruning = pruning
        self.pruning_type = pruning_type

        self.mode = mode
        self.plot_model = plot_model

        self.test_loss = []
        self.val_loss = []
        self.test_acc = []
        self.val_acc = []
        self.f1 = torch.tensor([0])
        self.conf_mat = torch.tensor([])
        self.epochs = 0
        self.fit_time = 0

        self.pruned_epoch = []
        self.times = []

        self.prune_time = 0


        if len(neurons) != number_of_layers:
            raise Exception("Each layer needs to have the number of neurons defined")

        # Define the network, as a sequential model alternatiing between Linear layers and ReLU layers.
        layers = OrderedDict()
        layers['input'] = nn.LazyLinear(neurons[0])  # initialize the input layer with the number o foutput features
        layers['relu_0'] = nn.ReLU()
        for i in range(1, number_of_layers - 1):
            layer_name = f"hidden_layer_{i}"
            layers[layer_name] = nn.Linear(neurons[i - 1], neurons[i])
            layer_name = f"relu_{i}"
            layers[layer_name] = nn.ReLU()
        layers['output'] = nn.Linear(neurons[number_of_layers - 2], neurons[number_of_layers - 1])

        super(DynamicNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(layers)
        self.loss = loss
        self.optimizer_func = optimizer
        self.learning_rate = learning_rate
        self.optimizer = self.optimizer_func(self.parameters(), lr=self.learning_rate)
        self.recently_added_layer = False  # stores if a layer is recently added in the network

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def remove_neurons(self):
        # copy the layer names and pruning masks
        layers = list(i[0] for i in self.linear_relu_stack.named_children() if not i[0].__contains__("relu"))
        removable_layers = layers[1:-1]
        # remove input and output layer because we cannot remove output neurons,
        # and im not sure the effects of removing input layers
        if len(removable_layers) == 0:
            return
        pruning_masks = list(i[1] for i in self.linear_relu_stack.named_buffers() if not i[0].__contains__("bias"))

        for i in range(len(removable_layers)):
            # get the columns and rows containing zeroes
            zeroes_col = torch.count_nonzero(getattr(self.linear_relu_stack, removable_layers[i]).weight, dim=0)
            zeroes_row = torch.count_nonzero(getattr(self.linear_relu_stack, removable_layers[i]).weight, dim=1)
            # the idea is to remove a neuron if it has no weights leaving it , or no weights arriving
            # this affects 3 layers
            # if a column of a matrix is all zeros, it means that there are no connections leaving the neuron,
            # so we can remove it from the previous matrix and the current one
            # if a row of a matrix is all zeros, it means that there are no connections arriving to the neuron,
            # so essentially the neurons it connects to just have a bias in the neuron,
            # so we can remove the neuron from the current matrix and the following.
            temp_weights_prev = getattr(self.linear_relu_stack, layers[i]).weight[zeroes_col != 0]
            temp_weights_curr = getattr(self.linear_relu_stack, layers[i + 1]).weight[zeroes_row != 0][:,
                                zeroes_col != 0]
            temp_weights_next = getattr(self.linear_relu_stack, layers[i + 2]).weight[:, zeroes_row != 0]

            # redefine the neurons in the layer
            self.neurons_per_layer[i + 1] = temp_weights_curr.shape[1]
            self.neurons_per_layer[i + 2] = temp_weights_curr.shape[0]
            # fix the pruning mask
            if pruning_masks:
                pruning_masks[i] = pruning_masks[i][zeroes_col != 0]
                pruning_masks[i + 1] = pruning_masks[i + 1][zeroes_row != 0][:, zeroes_col != 0]
                pruning_masks[i + 2] = pruning_masks[i + 2][:, zeroes_row != 0]

            # reinintialize the layer, and set the weight of the layers
            setattr(self.linear_relu_stack, layers[i],
                    nn.Linear(temp_weights_prev.shape[1], temp_weights_prev.shape[0]).to(self.device))
            setattr(getattr(self.linear_relu_stack, layers[i]), "weight", torch.nn.Parameter(temp_weights_prev))

            setattr(self.linear_relu_stack, layers[i + 1],
                    nn.Linear(temp_weights_curr.shape[1], temp_weights_curr.shape[0]).to(self.device))
            setattr(getattr(self.linear_relu_stack, layers[i + 1]), "weight", torch.nn.Parameter(temp_weights_curr))

            setattr(self.linear_relu_stack, layers[i + 2],
                    nn.Linear(temp_weights_next.shape[1], temp_weights_next.shape[0]).to(self.device))
            setattr(getattr(self.linear_relu_stack, layers[i + 2]), "weight", torch.nn.Parameter(temp_weights_next))
            # set the pruing masks
            if pruning_masks:
                setattr(self.linear_relu_stack, layers[i],
                        (prune.custom_from_mask(getattr(self.linear_relu_stack, layers[i]), name='weight',
                                                mask=pruning_masks[i])))
                setattr(self.linear_relu_stack, layers[i + 1],
                        (prune.custom_from_mask(getattr(self.linear_relu_stack, layers[i + 1]), name='weight',
                                                mask=pruning_masks[i + 1])))
                setattr(self.linear_relu_stack, layers[i + 2],
                        (prune.custom_from_mask(getattr(self.linear_relu_stack, layers[i + 2]), name='weight',
                                                mask=pruning_masks[i + 2])))

        # complete a similar task for the ouput layer, because there might be a neuron with no connections to the output
        zeroes_col = torch.count_nonzero(self.linear_relu_stack.output.weight, dim=0)
        temp_weights_prev = getattr(self.linear_relu_stack, removable_layers[-1]).weight[zeroes_col != 0]
        temp_weights_curr = self.linear_relu_stack.output.weight[:, zeroes_col != 0]
        if pruning_masks:
            pruning_masks[-2] = pruning_masks[-2][zeroes_col != 0]
            pruning_masks[-1] = pruning_masks[-1][:, zeroes_col != 0]

        setattr(self.linear_relu_stack, removable_layers[-1],
                nn.Linear(temp_weights_prev.shape[1], temp_weights_prev.shape[0]).to(self.device))
        setattr(getattr(self.linear_relu_stack, removable_layers[-1]), "weight", torch.nn.Parameter(temp_weights_prev))

        setattr(self.linear_relu_stack, "output",
                nn.Linear(temp_weights_curr.shape[1], temp_weights_curr.shape[0]).to(self.device))
        setattr(self.linear_relu_stack.output, "weight", torch.nn.Parameter(temp_weights_curr))

        if pruning_masks:
            setattr(self.linear_relu_stack, "output",
                    (prune.custom_from_mask(self.linear_relu_stack.output, name='weight', mask=pruning_masks[-1])))
            setattr(self.linear_relu_stack, removable_layers[-1],
                    (prune.custom_from_mask(getattr(self.linear_relu_stack, removable_layers[-1]), name='weight',
                                            mask=pruning_masks[-2])))

        # fix the neurons per layer
        self.neurons_per_layer[-2] = temp_weights_curr.shape[1]
        self.neurons_per_layer[-1] = temp_weights_curr.shape[0]
        # reset the optimizer
        self.optimizer = self.optimizer_func(self.parameters(), lr=self.learning_rate)
        del temp_weights_prev
        del temp_weights_curr
        del temp_weights_next
        torch.cuda.empty_cache()

    def train_loop(self, dataloader):

        self.batch.reset(len(dataloader))

        data_load_time = 0
        forward_time = 0
        loss_time = 0
        backprop_time = 0
        optimizer_time = 0
        time_enumeration = 0

        start_time = time.time()
        t6 = time.time()

        for batch, (X, y) in enumerate(dataloader):

            t1 = time.time()
            time_enumeration += t1 - t6

            self.optimizer.zero_grad()
            X, y = X.to(self.device), y.to(self.device)
            t2 = time.time()

            # Compute prediction and loss / reg
            pred = self.forward(X)
            t3 = time.time()
            loss = self.loss(pred, y)
            t4 = time.time()

            loss.backward()
            t5 = time.time()
            # optimize
            self.optimizer.step()

            t6 = time.time()

            data_load_time += t2 - t1
            forward_time += t3 - t2
            loss_time += t4 - t3
            backprop_time += t5 - t4
            optimizer_time += t6 - t5
            # progress bar
            self.batch.update()
        end_time = time.time()

        if self.mode != "script":
            print(f"epoch statistics: enumeration {time_enumeration:.2f}, "
                  f"data_load {data_load_time:.2f}, "
                  f"forward_time {forward_time:.2f}, "
                  f"loss_time {loss_time:.2f}, "
                  f"backprop_time {backprop_time:.2f}, "
                  f"optimizer_time {optimizer_time:.2f}")

        self.times.append([time_enumeration, data_load_time, forward_time, loss_time, backprop_time, optimizer_time])
        return end_time - start_time

    def test_loop(self, dataloader, test=True):

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        reg = 0

        pred_arr = []
        correct_arr = []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.forward(X)
                pred_arr.append(pred.argmax(1))
                correct_arr.append(y)
                test_loss += self.loss(pred, y).cpu()

                # test_loss += self.loss(pred, y).item() + 0.01*self.custom_regularization().item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        if self.mode != "script":
            if test:
                print(f"TEST: Accuracy: {(100 * correct):>0.1f}%,"
                      f" Avg loss: {test_loss:>8f}")
            else:
                print(f"VALIDATION: Accuracy: {(100 * correct):>0.1f}%, "
                      f"Avg loss: {test_loss:>8f}")

        return round(correct * 100, 2), test_loss.item()

    def fit(self, epochs, train_dataloader, test_dataloader, validation_dataloader):
        t1 = time.time()
        self.epoch = tqdm(desc="epochs", unit='epochs', position=0, leave=True, colour='#0000ff')
        self.batch = tqdm(desc="batches", unit='batch', position=1, leave=None, colour='#00cc66')
        parameters_to_prune = []
        named_layers = []

        loss_grad_val = []
        accuracy_grad_val = []

        x = []
        x_temp = 1
        # variable to keep track of when the network was changed, to prevent pruning from happening too often
        recently_changed_net = 0

        self.epochs += epochs
        self.epoch.reset(epochs)
        for t in range(epochs):
            if self.mode != "script":
                print(f"Epoch {t + 1}:")

            epoch_time = self.train_loop(train_dataloader)
            if self.mode != "script":
                print(f"trained in {epoch_time:.2f}s")
            accuracy_temp, loss_temp = self.test_loop(validation_dataloader, test=False)

            self.epoch.set_postfix({'accuracy': f"{accuracy_temp}%", 'loss': loss_temp})
            self.val_loss.append(loss_temp)
            self.val_acc.append(accuracy_temp)

            accuracy_temp, loss_temp = self.test_loop(test_dataloader, test=True)
            self.test_loss.append(loss_temp)
            self.test_acc.append(accuracy_temp)


            if t > 0 and t + 1 != epochs:
                loss_grad_val.append(np.diff(self.val_loss[-2:]))
                accuracy_grad_val.append(np.diff(self.val_acc[-2:]))

                if recently_changed_net == 0:
                    # execute pruing when gradient is good
                    # if values are positive becauese both grad is negative
                    if min(loss_grad_val) < 0:  # ensure the gradient of the loss is negative
                        # otherwise unintended execution will happen.
                        if (self.pruning_min <= loss_grad_val[-1] / min(
                                loss_grad_val) <= self.pruning_max and self.pruning == "dynamic") or \
                                (t % self.pruning_iter == 0 and self.pruning == "static"):
                            self.pruned_epoch.append(t + 1)
                            t3 = time.time()
                            # get the layers that pruning can be applied on (all layers that arent relu)
                            named_layers = (list(
                                i[0] for i in self.linear_relu_stack.named_children() if not i[0].__contains__("relu")))
                            num_vars = 0
                            for i in named_layers:
                                # set the pruning parameters to the weight matrices, not including biases
                                parameters_to_prune.append((getattr(self.linear_relu_stack, i), 'weight'))
                                num_vars += torch.sum(getattr(self.linear_relu_stack, i).weight != 0)
                            # set the pruning rate based on the number of variables. The more variables hte higher rate
                            if self.pruning_rate == "dynamic":
                                pruning_rate = (math.log(num_vars) - 6) / 10
                            else:
                                pruning_rate = self.pruning_rate
                            # enforce upper and lower bounds to the pruning rate
                            if pruning_rate <= 0:
                                pruning_rate = 0
                                if self.mode != "script":
                                    print(f"Current paramters: {num_vars}, maximum pruning reached.")
                            if pruning_rate > 0.9:
                                pruning_rate = 0.9
                                if self.mode != "script":
                                    print(f"Setting pruning rate to 0.9, model has over 3 million parameters,"
                                      f" and an uppder bound of pruning is enforced.")
                            # prune the model
                            torch.cuda.empty_cache()
                            if self.pruning_type == "l1":
                                prune.global_unstructured(
                                    parameters_to_prune,
                                    pruning_method=prune.L1Unstructured,
                                    amount=pruning_rate,
                                )
                            elif self.pruning_type == "random":
                                prune.global_unstructured(
                                    parameters_to_prune,
                                    pruning_method=prune.RandomUnstructured,
                                    amount=pruning_rate,
                                )

                            # remove any stray neurons
                            self.remove_neurons()
                            t4 = time.time()
                            self.prune_time += (t4 - t3)
                            if self.mode != "script":
                                print("--------------------------------------------")
                                print(f"pruned: {round(pruning_rate * 100, 2)}%")
                                print(f"pruning takes {t4 - t3}s")
                                print("--------------------------------------------")
                            if self.pruning == "dynamic":
                                recently_changed_net = 2
                            elif self.pruning == "static":
                                recently_changed_net = 0
                else:
                    recently_changed_net -= 1

            x.append(x_temp)
            x_temp += 1
            self.epoch.update()
        if self.mode != "script":
            print("--------------------------------------------")

            if self.pruning:
                print("PRUNING SUMMARY:")
                curr_parameters = 0
                total_parameters = 0
                for i in named_layers:
                    temp1 = torch.sum(getattr(self.linear_relu_stack, i).weight == 0)
                    temp2 = float(getattr(self.linear_relu_stack, i).weight.nelement())
                    curr_parameters += temp1
                    total_parameters += temp2
                    sparsity = 100. * float(temp1 / temp2)
                    print(f"Sparsity in {i}: {temp1}/{temp2} = {round(sparsity, 2)}%")
                if total_parameters != 0:
                    print(
                        f"Total Sparsity in the whole model: {round(100. * float(curr_parameters / total_parameters), 2)}%")
                    print(
                        f"Orignally there were {total_parameters} weights, now there are {total_parameters - curr_parameters} weights")
                else:
                    print("No parameters were pruned.")
            print("--------------------------------------------")
            print(f"Pruned Neurons on epochs: {self.pruned_epoch}")

        t2 = time.time()
        self.fit_time = t2 - t1

        if self.plot_model:

            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True)
            # TODO: fix the graphs
            x_grad = [i + 0.5 for i in range(1, len(x))]
            ax[0][0].plot(x, self.val_loss, label="Validation")
            ax[0][0].plot(x, self.test_loss, label="Test")
            ax[0][0].set_title("Model Loss")
            ax[0][0].set_ylabel("Loss")
            ax[0][0].set_xlabel("Epoch")
            ax[0][0].xaxis.set_tick_params(which='both', labelbottom=True)
            ax[1][0].plot(x_grad, np.diff(self.val_loss) / min(np.diff(self.val_loss)), label="Validation")
            ax[1][0].plot(x_grad, np.diff(self.test_loss) / min(np.diff(self.test_loss)), label="Test")
            ax[1][0].set_title("Gradient of Model Loss")
            ax[1][0].set_ylabel("Relative Gradient of Loss")
            ax[1][0].set_xlabel("Epoch")
            ax[0][1].plot(x, self.val_acc, label="Validation")
            ax[0][1].plot(x, self.test_acc, label="Test")
            ax[0][1].set_title("Model Accuracy")
            ax[0][1].set_ylabel("Accuracy")
            ax[0][1].set_xlabel("Epoch")
            ax[0][1].xaxis.set_tick_params(which='both', labelbottom=True)
            ax[1][1].plot(x_grad, np.diff(self.val_acc), label="Validation")
            ax[1][1].plot(x_grad, np.diff(self.test_acc), label="Test")
            ax[1][1].set_title("Gradient of Model Accuracy")
            ax[1][1].set_ylabel("Gradient of Accuracy")
            ax[1][1].set_xlabel("Epoch")
            plt.show()
        self.test(test_dataloader)
        return self.test_loop(test_dataloader)

    def test(self, dataloader):
        pred_arr = []
        correct_arr = []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.forward(X)
                pred_arr.append(pred.argmax(1))
                correct_arr.append(y)

        confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.neurons_per_layer[-1]).to(self.device)
        f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.neurons_per_layer[-1]).to(self.device)

        if self.mode != "script":
            print("Confusion Matrix:")
            print(confmat(torch.cat(pred_arr), torch.cat(correct_arr)))
            print(f"f1:{f1(torch.cat(pred_arr), torch.cat(correct_arr))}")

        self.f1 = f1(torch.cat(pred_arr), torch.cat(correct_arr))
        self.conf_mat = confmat(torch.cat(pred_arr), torch.cat(correct_arr))

        return

    def visualize(self, input_shape=1):
        self.neurons_per_layer[0] = input_shape
        graph_visualizer(self.number_of_layers, self.neurons_per_layer, (list(
            getattr(self.linear_relu_stack, f"{i[0]}").weight for i in self.linear_relu_stack.named_children() if
            not i[0].__contains__("relu"))))
        return

    def save(self, base_dir="results/test"):
        model = '_'.join([str(elem) for elem in self.neurons_per_layer[1:]])
        timestamp = time.time()

        optimizer = type(self.optimizer).__name__
        param_list = [
            self.neurons_per_layer,
            optimizer, self.learning_rate,
            self.pruning, self.pruning_iter, self.pruning_rate, self.pruning_type, self.pruning_min, self.pruning_max,
            self.epochs
        ]

        params = [self.pruning, self.pruning_iter, self.pruning_rate, self.pruning_type]

        param_string = '_'.join([str(elem) for elem in params])
        dir_name = base_dir + "/" + model + "_" + param_string

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        torch.save(list(
            getattr(self.linear_relu_stack, f"{i[0]}").weight for i in self.linear_relu_stack.named_children() if
            not i[0].__contains__("relu")), dir_name + "/" + str(int(timestamp)) + ".weight")
        summaries = {
            "test_loss": self.test_loss, "validation_loss": self.val_loss,
            "test_accruacy": self.test_acc, "validation_accruacy": self.val_acc,
            "f1": self.f1.item(), "confusion_matrix": self.conf_mat.tolist(),
            "total_time": self.fit_time, "train_time": self.times, "prune_time": self.prune_time,
            "pruned_epoch": self.pruned_epoch,
            "parameters": param_list
        }

        with open(dir_name + "/" + str(int(timestamp)) + ".json", 'w+') as file:
            file.write(json.dumps(summaries))
            file.close()
