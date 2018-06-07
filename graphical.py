import matplotlib.pyplot as plt

def display_evolution(ff_by_generation : list, title : str):
    plt.plot(list(range(len(ff_by_generation))), ff_by_generation)
    #plt.interactive(True)
    plt.title(title)
    plt.show()