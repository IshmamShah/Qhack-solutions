#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #

    metric = np.zeros((len(params),len(params)))
    gradient = np.zeros(len(params))
    
    psi_p = statev(params)
    
    for i in range(len(params)):
        for j in range(len(params)):
            theta1 = params.copy()
            theta2 = params.copy()
            theta3 = params.copy()
            theta4 = params.copy()
            
            theta1[i] = theta1[i] + np.pi/2
            theta1[j] = theta1[j] + np.pi/2
            
            theta2[i] = theta2[i] + np.pi/2
            theta2[j] = theta2[j] - np.pi/2
            
            theta3[i] = theta3[i] - np.pi/2
            theta3[j] = theta3[j] + np.pi/2

            theta4[i] = theta4[i] - np.pi/2
            theta4[j] = theta4[j] - np.pi/2
            
            psi1 = statev(theta1)
            psi2 = statev(theta2)
            psi3 = statev(theta3)
            psi4 = statev(theta4)
            
            s1 = abs(np.conjugate(psi_p) @ psi1)**2
            s2 = abs(np.conjugate(psi_p) @ psi2)**2
            s3 = abs(np.conjugate(psi_p) @ psi3)**2
            s4 = abs(np.conjugate(psi_p) @ psi4)**2
            metric[i,j] = (1/8) * (-s1 + s2 + s3 - s4)
            
            
    met_inv = np.linalg.inv(metric)
    

#     print(met_inv)
            
    for i in range(len(params)):
        gradient[i] = gradnt(i)

    natural_grad = np.dot(met_inv,gradient)
            
    # QHACK #

    return natural_grad

@qml.qnode(dev)
def statev(params):
    variational_circuit(params)
    return (qml.state())

def gradnt(c1):
    s = 0.5
    theta1 = params.copy()
    theta2 = params.copy()
    theta1[c1] = theta1[c1] + s*np.pi 
    theta2[c1] = theta2[c1] - s*np.pi 

    r_plus = qnode(theta1)
    r_minus = qnode(theta2)
    return 0.5 * (r_plus - r_minus)

def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
