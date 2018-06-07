import math

def function_sigmoid(inp):
    try:
        return 1 / (1 + math.exp(-inp))
    except OverflowError:
        print('An approximation has been made during the computing of the sigmoid function.')
        return 0

def function_step(inp):
    """
    The transfer function needed for the perceptron.
    :param inp: x
    :return:    f(x)
    """
    if inp > 0:
        return float(1)
    else:
        return float(0)

def function_linear(inp):
    return inp


#=======================================================================================================================

def mean_square_error(list1 : list, list2 : list):
    """
    Not a function for the NN like the other in this file. Meant to be used by the individual to calcul his FF
    :param list1: 
    :param list2: 
    :return: mean square error
    """
    result = 0
    for i, a in enumerate(list1):
        result += (list2[i] - a)**2
    return result

