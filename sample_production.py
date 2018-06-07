import csv
import sys
import math
import numpy

class sample:
    """
    Simple sample class containing only the parameter and the category of the sample.
    """
    def __init__(self, crit=[], wanted_output=None):
        self.crit = crit
        self.wanted_output = wanted_output

    def __str__(self):
        return 'Crit : ' + str(self.crit) + '\nClass : ' + str(self.wanted_output)

def read_data_obj(filename, number_of_output):
    """
    This function reads the CSV file and return an array of sample object
    :param filename: name of the CSV file
    :return:samples : array of sample object
    """
    samples=[]
    try:
        with open(filename) as csfile:
            data_reader = csv.reader(csfile)
            for line in data_reader:
                crit = []
                for a in line[:-number_of_output]:
                    crit.append(float(a))
                output = []
                for b in line[-number_of_output:]:
                    output.append(float(b))
                samples.append(sample(crit=crit, wanted_output=output))
    except FileNotFoundError:
        print("Wrong filename !")
        input("Press enter to exit.")
        sys.exit()
    return samples

def generate_function_to_approximate(function, range : list, number_of_samples : int):
    x = numpy.linspace(range[0], range[1], number_of_samples)
    y = []
    for a in x:
        y.append(function(a))
    return x,y
