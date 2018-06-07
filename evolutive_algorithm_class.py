import random
from functions import mean_square_error
import numpy
import math


TAU = 1/math.sqrt(18)
TAU_PRIME = 1/math.sqrt(2*math.sqrt(9))


class population:
    def __init__(self,number_of_weights, range_of_weight, range_of_s, number_of_individuals : int = 20,  number_of_children : int = 40, accuracy : float = 0.01, max_generations : int = 200 ):
        """

        :param number_of_weights:
        :param range_of_weight:
        :param range_of_s:
        :param number_of_individuals:
        :param number_of_survivor:
        :param number_of_children:
        :param accuracy:
        :param max_generations:
        """
        self.number_of_children = number_of_children
        self.accuracy = accuracy
        self.number_of_survivor = number_of_individuals
        self.max_generations = max_generations
        self.population_of_individuals = []
        for i in range(number_of_individuals):
            self.population_of_individuals.append(individual(number_of_weights=number_of_weights, range_of_weight=range_of_weight, range_of_s=range_of_s))

    def selection(self, individuals : list):
        individuals.sort(key= lambda indi: indi.value_ff)
        return individuals[:self.number_of_survivor]

    def mutation(self, reseau):
        childs = []
        for i in range(self.number_of_survivor):
            childs.append(random.choice(self.population_of_individuals).mutation())
        for a in childs:
            a.value_ff = sum(a.fitness_function(reseau=reseau))
        return childs

    def calculate_global_mse(self):
        global_mse = 0
        for indi in self.population_of_individuals:
            global_mse += indi.value_ff
        return global_mse


class individual:
    def __init__(self, number_of_weights : int, range_of_weight : list, range_of_s : list, initialisation = True):
        self.value = []
        self.s_array = []
        if initialisation is True:
            for a in range(number_of_weights):
                self.value.append(random.uniform(range_of_weight[0],range_of_weight[1]))

            for b in range(number_of_weights):
                self.s_array.append(random.uniform(range_of_s[0],range_of_s[1]))
        self.value_ff = 0

    def __str__(self):
        return 'Weights value : ' + str(self.value) + '\nS value : ' +str(self.s_array) + '\n FF :' + str(self.value_ff)

    def fitness_function(self, reseau):
        reseau.change_weights(self.value)
        indi_output = []
        for a in reseau.produce_output():
            indi_output.append(a)
        result = []

        for i, a in enumerate(indi_output):
            result.append(mean_square_error(a, reseau.goal_output[i]))
        #print('FF of indi :', result)
        return result

    def mutation(self):
        rng_indi_s = numpy.random.normal(0,1)
        rng_indi_w = numpy.random.normal(0,1)
        new_s = []
        for ses in self.s_array:
            new_s.append(ses*math.exp(TAU_PRIME*rng_indi_s + TAU*numpy.random.normal(0,1)))
        new_w = []
        for i, wew in enumerate(self.value):
            new_w.append(wew + new_s[i] * rng_indi_w)
        new_indi = individual(0,[],[],initialisation=False)
        new_indi.s_array = new_s
        new_indi.value = new_w
        return new_indi             #CALCULER LA FF QUELQUEPART !!!!!!!!!!!!!!!!!!!!













