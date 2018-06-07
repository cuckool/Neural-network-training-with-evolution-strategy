import random
import csv
import sys
from evolutive_algorithm_class import *
import progressbar

class neural_network:
    def __init__(self, training_data : list, layer_structure : list, function_by_layer : list, **options):
        """

        :param training_data: list of the samples used for training
        :param layer_structure:
        :param function_by_layer:
        :param number_of_inputs:
        :param options: Available options:

        Regarding the initial value of the weights :

        - value : starting value for all the weights
        - range : range in which we generate a random number as starting value for each weight
        If none of these options are used the weights will be randomly initialised between [-10, 10]

        Regarding the parameter of the population:

        range_of_weight = range between which the weights are initiated (ex : [-5, 5])
        range_of_s = range between which the s are initiated (ex : [-5, 5])
        number_of_individuals = (int) the number of individuals of the population
        number_of_children = (int) the number of children created by mutation
        accuracy = the minimum accuracy that we have to reach to stop the recursion
        max_generations = the maximum generation, after this limit the algorithm will stop the recursion

        """
        if len(layer_structure) != len(function_by_layer):
            print('Error in the dimensions of the parameter : the number of functions is different from the number of layers.')
            exit()
        if 'value' in options and 'range' in options:
            print('Error : the value option and the range options are incompatible. Use only one of them.')
            exit()
        self.inputs = len(training_data[0].crit)
        self.training_data = training_data
        self.goal_output = []
        self.initialized = False
        for a in self.training_data:
            self.goal_output.append(a.wanted_output)
        self.function_by_layer = function_by_layer



        self.structure = []
        number_neuron_of_preceding_layer = self.inputs
        for i in range(len(layer_structure)):
            layer = []
            for j in range(1, layer_structure[i]+1):
                weights = []
                for k in range(1,number_neuron_of_preceding_layer+1):
                    weights.append(weight(id_root_neuron=[i, k], id_arrival_neuron=[i+1, j], **options))
                layer.append(neuron(transfer_function=function_by_layer[i], id_neuron=j, id_layer=i+1, weights=weights, **options))
            number_neuron_of_preceding_layer = len(layer)
            self.structure.append(layer)
                                                            #structure : [[neuron 11, neuron 12],[neuron 21, neuron 22]]
        nb_weights = 0
        for i in self.structure:
            nb_weights += len(i[0].weights) * len(i)
        self.pop = population(number_of_weights=nb_weights,
                              range_of_weight=options.get('range_w', [-10, 10]),
                              range_of_s=options.get('range_s', [0.000001, 5]),
                              number_of_individuals=options.get('number_of_individuals',20),
                              number_of_children=options.get('number_of_children',40),
                              accuracy=options.get('accuracy',0.01),
                              max_generations=options.get('max_generations', 200))


    def __str__(self):
        string = ''
        for i in self.structure:
            string += ('===================================================================================================\n')
            for j in i:
                string += str(j) + '\n'
        string += ('===================================================================================================\n')
        return string


    def train(self):
        #creating fitness score for the initial pop
        if self.initialized is False:
            for a in self.pop.population_of_individuals:
                a.value_ff = sum(a.fitness_function(self))
            self.initialized = True
        generation = 0
        success = False
        global_ff_by_generations = []
        best_ff_by_generations = []
        #entering the main loop
        with progressbar.ProgressBar(max_value=self.pop.max_generations) as bar:
            while success is False:
                bar.update(generation)
                global_ff_by_generations.append(self.pop.calculate_global_mse())
                if self.pop.population_of_individuals[0].value_ff < self.pop.accuracy:
                    success = True
                if generation == self.pop.max_generations:
                    break
                childs = self.pop.mutation(self)
                parents = self.pop.population_of_individuals
                self.pop.population_of_individuals = self.pop.selection(childs+parents)
                best_ff_by_generations.append(self.pop.population_of_individuals[0].value_ff)
                generation += 1
            return self.pop.population_of_individuals[0], success, generation, global_ff_by_generations, best_ff_by_generations


    def produce_output(self, dataset=None):
        """
        Use the network to produce the output for the dataset
        :param dataset: if None, the network will use the training dataset it was created with
        :return: output for each sample
        """
        if dataset:
            samples = dataset
        else:
            samples = self.training_data

        for sample in samples:
            input_to_neuron = sample.crit
            for layer in self.structure:
                output = []
                for neurone in layer:
                    #print(neurone.id_layer,neurone.id_neuron)
                    #print('input given by network',input_to_neuron)
                    output.append(neurone.get_output(input_to_n=input_to_neuron))
                    #print('output :',output)
                input_to_neuron = output
            yield output


    def change_weights(self,  arr_weights : list):
        nb_weights_by_layer = [0]
        for i in self.structure:
            nb_weights_by_layer.append(len(i[0].weights) * len(i))  # number of weight in layer = nb of neuron * nb of weight in a neuron
        if sum(nb_weights_by_layer) != len(arr_weights):
            print('Error, the number of weights given is not equal to the number of weights needed !')
            exit()

        partitioned_arr_weights = []        #One slice > the whole weights for 1 layer

        #print('given weights', arr_weights)
        #print('nb weight by layer', nb_weights_by_layer)
        for i in range(len(nb_weights_by_layer)-1):
            partitioned_arr_weights.append(arr_weights[nb_weights_by_layer[i]:nb_weights_by_layer[i] + nb_weights_by_layer[i+1]])
        #print(partitioned_arr_weights)

        for b in range(len(partitioned_arr_weights)):
            for a in range(0, len(partitioned_arr_weights[b]), len(self.structure[b][0].weights)):
                #print (a)
                self.structure[b][int(a/len(self.structure[b][0].weights))].change_weight_neuron(partitioned_arr_weights[b][a:a+len(self.structure[b][0].weights)])



class neuron:
    def __init__(self, transfer_function, id_neuron : int, id_layer : int, weights : list, **options):
        """

        :param transfer_function:
        :param id_neuron:
        :param id_layer:
        :param weights:
        :param options: Available options :
        - value : default value for the first weight
        - range : range in which we generate a random number as starting value for each weight
        """
        self.transfer_function = transfer_function
        self.id_neuron = id_neuron
        self.id_layer = id_layer
        self.weights = weights
        self.weights.insert(0, weight(id_root_neuron=[-1,-1], id_arrival_neuron=[self.id_layer, self.id_neuron], **options))        #weight du début, sans input (b ou w0)

    def __str__(self):
        """
        Numérotation neuron : [id_layer,id_neuron]
        :return:
        """
        string =''
        string += 'Neuron ' + str(self.id_neuron) + ' | layer ' + str(self.id_layer) + '\n'
        for i in self.weights:
            string += '     ' + str(i) +'\n'
        return string

    def get_output(self, input_to_n : list):
        result=0
        crit = list(input_to_n)
        crit.insert(0,1.0)
        for a in range(len(crit)):
            result += crit[a] * self.weights[a].value
        result = self.transfer_function(result)
        return result

    def change_weight_neuron(self, weight_arr : list):
        for i, wei in enumerate(self.weights):
            wei.value = weight_arr[i]


class weight:
    def __init__(self, id_root_neuron : list, id_arrival_neuron : list, **options):
        """
        :param id_root_neuron:
        :param id_arrival_neuron:
        :param options: list of options available:
        - value : give default starting value
        - range : range for the random generation of starting value
        """
        self.id = []
        self.id.append(id_root_neuron)
        self.id.append(id_arrival_neuron)
        if options.get('value'):
            self.value = options['value']
        else:
            self.value = random.uniform(options.get('range',[-2,2])[0], options.get('range',[-2,2])[1])

    def __str__(self):
        """
        Numérotation weight: [id_root_neuron,id_arrival_neuron] : [ [id_layer,id_neuron] , [id_layer,id_neuron] ]
        :return:
        """
        return 'W : L' + str(self.id[0][0]) +' N' + str(self.id[0][1]) + ' ==> ' + 'L' + str(self.id[1][0]) +' N' + str(self.id[1][1]) + ' \t| ' + str(self.value)



