#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    
#     def gradnt(c1):
#         s = 0.5
#         theta1 = weights.copy()
#         theta2 = weights.copy()
#         theta1[c1] = theta1[c1] + s*np.pi 
#         theta2[c1] = theta2[c1] - s*np.pi 
        
#         r_plus = circuit(theta1)
#         r_minus = circuit(theta2)
#         return 0.5 * (r_plus - r_minus)
        
    def param_shift(c1,c2):
    # using the convention u=1/2
        s = 0.5
        theta1 = weights.copy()
        theta2 = weights.copy()
        theta3 = weights.copy()
        theta4 = weights.copy()
        
        theta1[c1] = theta1[c1] + s*np.pi 
        theta1[c2] = theta1[c2] + s*np.pi 
        theta2[c1] = theta2[c1] + s*np.pi 
        theta2[c2] = theta2[c2] - s*np.pi 
        theta3[c1] = theta3[c1] - s*np.pi 
        theta3[c2] = theta3[c2] + s*np.pi 
        theta4[c1] = theta4[c1] - s*np.pi 
        theta4[c2] = theta4[c2] - s*np.pi 
        
        r_1 = circuit(theta1)
        r_2 = circuit(theta2)
        r_3 = circuit(theta3)
        r_4 = circuit(theta4)

        return 0.25 *( r_1 - r_2 - r_3 + r_4 )
    
    theta_r = weights.copy()
    theta_r = theta_r
    r_d = circuit(theta_r)
    
    def hess_diag(c1):
        s = 0.5
        theta1 = weights.copy()
        theta2 = weights.copy()
        theta1 = theta1
        theta2 = theta2
        theta1[c1] = theta1[c1] + s*np.pi 
        theta2[c1] = theta2[c1] - s*np.pi
        r_1 = circuit(theta1)
        r_2 = circuit(theta2)
        
        return 0.25*(2*r_1 - 4*r_d + 2*r_2), 0.5*(r_1 - r_2)
        
    for i in range(5):
        hessian[i,i], gradient[i] = hess_diag(i)
        for j in range(i+1,5):
            hessian[i,j] = param_shift(i,j)
            hessian[j,i] = hessian[i,j]
    
    
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)
    
    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
