from evolutive_algorithm_class import *
from neural_network import *
from functions import *
from sample_production import *
import graphical

samples = read_data_obj(filename=r'C:\Users\Crowbar\PycharmProjects\BIA\trainNNwithEvolutiveAlgo\samples.txt', number_of_output=1)
reseau = neural_network(training_data=samples, function_by_layer=[function_sigmoid, function_sigmoid], layer_structure=[2, 1], max_generations=2000, range_w=[-10, 10])
print(reseau)

results = reseau.train()
print('Success :', results[1])
print('Nb of generation :', results[2])
print('Best individual :\n', results[0])
weight_output = []
for a in reseau.produce_output():
    weight_output.append(a)
print('Output for this weight :',weight_output)
print('Ideal output :', reseau.goal_output)

graphical.display_evolution(results[3], 'Evolution of the global MSE for each generation')
graphical.display_evolution(results[4], 'Evolution of the best FF for each generation')



#indi = individual(number_of_weights=7, range_of_s=[0,5], range_of_weight=[-5,5])
#print(indi.fitness_function(reseau = reseau))
"""
nb_weights_by_layer = []
for i in reseau.structure:
    nb_weights_by_layer.append(len(i[0].weights) * len(i))
print(nb_weights_by_layer)
new_weights = [1,1,1,1,1,1,1,1,1,1,3,3,1,1,1]
print('size new weights', len(new_weights))
reseau.change_weights_V2(new_weights)
print(reseau)
"""

"""
pop = population(number_of_weights=2, range_of_weight=[-1,1], range_of_s=[0.00005,5])
pop.population_of_individuals[0]
"""
"""
wei = weight(id_arrival_neuron=[0,0], id_root_neuron=[0,0], value=55)
neu = neuron(transfer_function=function_sigmoid, id_layer=0, id_neuron=0, value = 1, weights=[wei])
print(neu)
neu.change_weight_neuron(weight_arr=[1,1])
print(neu)
"""