#!/usr/bin/env python
# coding: utf-8

# In[3]:


#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []

    # QHACK #
    from pennylane import qaoa
    wires = range(6)
    dev = qml.device('default.qubit', wires=6)
    cost_h, mixer_h = qaoa.max_independent_set(graph, constrained=True)
    
    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)
    
    def circuit(params, **kwargs):        
        qml.layer(qaoa_layer, N_LAYERS, params[0], params[1])
    
    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=wires)


    probs = probability_circuit(params[0], params[1])
    def heavy_output_set(m, probs):
        # Compute heavy outputs of an m-qubit circuit with measurement outcome
        # probabilities given by probs, which is an array with the probabilities
        # ordered as '000', '001', ... '111'.

        # Sort the probabilities so that those above the median are in the second half
        probs_ascending_order = np.argsort(probs)
        sorted_probs = probs[probs_ascending_order]

        # Heavy outputs are the bit strings above the median
        heavy_outputs = [
            # Convert integer indices to m-bit binary strings
            format(x, f"#0{m+2}b")[2:] for x in list(probs_ascending_order[2 ** (m - 1) :])
        ]

        # Probability of a heavy output
        prob_heavy_output = np.sum(sorted_probs[2 ** (m - 1) :])

        return heavy_outputs, prob_heavy_output
    heavy_outputs, prob_heavy_output = heavy_output_set(6, probs)
    max_ind_set = []
    for i in range(len(heavy_outputs[-1])):
        if heavy_outputs[-1][i] == '1':
            max_ind_set.append(i)

    # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)
    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)


# In[ ]:




