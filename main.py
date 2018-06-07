from neural_network import *
from functions import *
from sample_production import *
import graphical
import sys

#######################################################################################################
################## MANDATORY PARAMETER - MANDATORY PARAMETER - MANDATORY PARAMETER ####################
#######################################################################################################

#If you want to change the structure of the network, the range of the weights and many other parameter, you should look
#into this section and change the variable (with respect to the rule next to them).

#TRANSFER FUNCTION for each layer :
#Rule : must be a list with the same lenght as the 'layer_structure' list. The available are the methods beginning by
# 'function' that are located in functions.py, the first entry will be the tranfer function for the first layer and
# so on...
FUNCTION_TO_USE_BY_LAYER = [function_sigmoid, function_sigmoid]

#LAYER STRUCTURE for the neural network:
#Rule : must be a list with the same lenght as the 'function_to_use_by_layer' list. The first entry determine the number
# of neuron for the first layer, the second the number of neuron for the second layer and so on. The length of this list
# indicate the number of layer.
LAYER_STRUCTURE = [2, 1]

#MAX GENERATIONS for the training:
#Rule : must be an integer. The number of maximum generation for the training of the neural network.
MAX_GENERATIONS = 5000

####################################################################################################
################## OPTIONAL PARAMETER - OPTIONAL PARAMETER - OPTIONAL PARAMETER ####################
####################################################################################################

#To use the following parameters, you have to add them to the creation of the 'reseau' object. For example, if I want to
# use the value option :
#reseau = neural_network(training_data=samples,
#                        function_by_layer=FUNCTION_TO_USE_BY_LAYER,
#                        layer_structure=LAYER_STRUCTURE,
#                        max_generations=MAX_GENERATIONS,
#                        value = VALUE)

#VALUE OF WEIGHTS for the evolutive algorithm:
#Rule : must be an int. All the weights will be initialized at this value
VALUE = 1                   #default : not used by default

#RANGE OF WEIGHTS for the evolutive algorithm:
#Rule : must be a list of 2 entry. The weights will be initialized between the first and the last entry.
# THIS OPTION CANNOT BE USED WITH the 'VALUE' option.
RANGE_W = [-10, 10]         #default : [10,10]

#RANGE OF S for the evolutive algorithm:
#Rule : must be a list of 2 entry. The s will be initialized between the first and the last entry.
RANGE_S = [0.000001, 5]          #default : [0.000001, 5]

#NUMBER OF INDIVIDUALS in the population:
#Rule : must be a positive int. This is the number of individual in the population ( mu )
NUMBER_OF_INDIVIDUALS = 40  #default : 20

#NUMBER OF CHILDREN for the mutations:
#Rule : must be a positive int, this is the number of individuals that will be created for the mutation phase ( lambda )
NUMBER_OF_CHILDREN = 80     #default : 40

#ACCURACY :
#Rule : must be a float, the minimum accuracy at which the algorithm will stop iteratig (represent a minimum fitness,
#that has to be reached by the best individual).
ACCURACY = 0.01              #default : 0.01

if len(sys.argv) != 3:
    print('Please notice that this program need 2 arguments, the first being the path to the csv file containing the '
          'samples. The second being the number of output of these sample.')
    exit()

samples = read_data_obj(filename=sys.argv[1], number_of_output=int(sys.argv[2]))
reseau = neural_network(training_data=samples,
                        function_by_layer=FUNCTION_TO_USE_BY_LAYER,
                        layer_structure=LAYER_STRUCTURE,
                        max_generations=MAX_GENERATIONS,
                        number_of_individuals=NUMBER_OF_INDIVIDUALS,
                        number_of_children=NUMBER_OF_CHILDREN)
print(reseau)

#RESULTS OUTPUT :
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