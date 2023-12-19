import random
from random import randrange
from time import time 

from tkinter import *
import tkinter as tk
from tkinter import ttk

class Problem_Genetic(object):
    
    def __init__(self,genes,individuals_length,decode,fitness):
        self.genes= genes
        self.individuals_length= individuals_length
        self.decode= decode
        self.fitness= fitness

    def mutation(self, chromosome, prob):
            
            def inversion_mutation(chromosome_aux):
                chromosome = chromosome_aux
                
                index1 = randrange(0,len(chromosome))
                index2 = randrange(index1,len(chromosome))
                
                chromosome_mid = chromosome[index1:index2]
                chromosome_mid.reverse()
                
                chromosome_result = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
                
                return chromosome_result
        
            aux = []
            for _ in range(len(chromosome)):
                if random.random() < prob :
                    aux = inversion_mutation(chromosome)
            return aux

    def crossover(self,parent1, parent2):

        def process_gen_repeated(copy_child1,copy_child2):
            count1=0
            for gen1 in copy_child1[:pos]:
                repeat = 0
                repeat = copy_child1.count(gen1)
                if repeat > 1:#If need to fix repeated gen
                    count2=0
                    for gen2 in parent1[pos:]:#Choose next available gen
                        if gen2 not in copy_child1:
                            child1[count1] = parent1[pos:][count2]
                        count2+=1
                count1+=1

            count1=0
            for gen1 in copy_child2[:pos]:
                repeat = 0
                repeat = copy_child2.count(gen1)
                if repeat > 1:#If need to fix repeated gen
                    count2=0
                    for gen2 in parent2[pos:]:#Choose next available gen
                        if gen2 not in copy_child2:
                            child2[count1] = parent2[pos:][count2]
                        count2+=1
                count1+=1

            return [child1,child2]

        pos=random.randrange(1,self.individuals_length-1)
        child1 = parent1[:pos] + parent2[pos:] 
        child2 = parent2[:pos] + parent1[pos:] 
        
        return  process_gen_repeated(child1, child2)
    
   
def decodeVRP(chromosome):    
    list=[]
    for (k,v) in chromosome:
        if k in trucks[:(num_trucks-1)]:
            list.append(frontier)
            continue
        list.append(cities.get(k))
    return list


def penalty_capacity(chromosome):
        actual = chromosome
        value_penalty = 0
        capacity_list = []
        index_cap = 0
        overloads = 0
        
        for i in range(0,len(trucks)):
            init = 0
            capacity_list.append(init)
            
        for (k,v) in actual:
            if k not in trucks:
                capacity_list[int(index_cap)]+=v
            else:
                index_cap+= 1
                
            if  capacity_list[index_cap] > capacity_trucks:
                overloads+=1
                #penalty 
                value_penalty+= 100 * overloads
        return value_penalty

def fitnessVRP(chromosome):
    
    def distanceTrip(index,city):
        w = distances.get(index)
        return  w[city]
        
    actualChromosome = chromosome
    fitness_value = 0
   
    penalty_cap = penalty_capacity(actualChromosome)
    for (key,value) in actualChromosome:
        if key not in trucks:
            nextCity_tuple = actualChromosome[key]
            if list(nextCity_tuple)[0] not in trucks:
                nextCity= list(nextCity_tuple)[0]
                #fitness
                fitness_value+= distanceTrip(key,nextCity) + (50 * penalty_cap)
    return fitness_value




def genetic_algorithm_t(Problem_Genetic,k,opt,ngen,size,ratio_cross,prob_mutate):
    
    def initial_population(Problem_Genetic,size):   
        def generate_chromosome():
            chromosome=[]
            for i in Problem_Genetic.genes:
                chromosome.append(i)
            random.shuffle(chromosome)
            return chromosome
        return [generate_chromosome() for _ in range(size)]
            
    def new_generation_t(Problem_Genetic,k,opt,population,n_parents,n_directs,prob_mutate):
        
        def tournament_selection(Problem_Genetic,population,n,k,opt):
            winners=[]
            for _ in range(n):
                elements = random.sample(population,k)
                winners.append(opt(elements,key=Problem_Genetic.fitness))
            return winners
        
        def cross_parents(Problem_Genetic,parents):
            childs=[]
            for i in range(0,len(parents),2):
                childs.extend(Problem_Genetic.crossover(parents[i],parents[i+1]))
            return childs
    
        def mutate(Problem_Genetic,population,prob):
            for i in population:
                Problem_Genetic.mutation(i,prob)
            return population
           
        directs = tournament_selection(Problem_Genetic, population, n_directs, k, opt)
        crosses = cross_parents(Problem_Genetic,
                                tournament_selection(Problem_Genetic, population, n_parents, k, opt))
        mutations = mutate(Problem_Genetic, crosses, prob_mutate)
        new_generation = directs + mutations
        
        return new_generation
    
    population = initial_population(Problem_Genetic, size)
    n_parents = round(size*ratio_cross)
    n_parents = (n_parents if n_parents%2==0 else n_parents-1)
    n_directs = size - n_parents
    
    for _ in range(ngen):
        population = new_generation_t(Problem_Genetic, k, opt, population, n_parents, n_directs, prob_mutate)
    
    bestChromosome = opt(population, key = Problem_Genetic.fitness)
    print("Chromosome: ", bestChromosome)
    genotype = Problem_Genetic.decode(bestChromosome)
    print ("Solution: " , (genotype,Problem_Genetic.fitness(bestChromosome)))
    genetic_algorithm_t.x=bestChromosome
    return (genotype,Problem_Genetic.fitness(bestChromosome))



def VRP(k):
    VRP_PROBLEM = Problem_Genetic([(0,10),(1,10),(2, 10),(3,10),(4,10),(5,10),(6,10),(7,10),
                                   (trucks[0],capacity_trucks)],
                                  len(cities), lambda x : decodeVRP(x), lambda y: fitnessVRP(y))
    
    def first_part_GA(k):
        cont  = 0
        print ("---------------------------------------------------------Executing FIRST PART: VRP --------------------------------------------------------- \n")
        print("Capacity of trucks = ",capacity_trucks)
        print("Frontier = ",frontier)
        print("")
        tiempo_inicial_t2 = time()
        while cont <= k: 
            genetic_algorithm_t(VRP_PROBLEM, 2, min, 200, 100, 0.8, 0.05)
            cont+=1
        tiempo_final_t2 = time()
        print("\n") 
        print("Total time: ",(tiempo_final_t2 - tiempo_inicial_t2)," secs.\n")
    
    
    first_part_GA(k)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")
#---------------------------------------- AUXILIARY DATA FOR TESTING --------------------------------

#CONSTANTS

cities = {0:'Almeria',1:'Cadiz',2:'Cordoba',3:'Granada',4:'Huelva',5:'Jaen',6:'Malaga',7:'Sevilla'}

# Distance between each pair of cities
# paireddistance=input("enter the distancebetween cities")

w0 = [999,454,317,165,528,222,223,410]
w1 = [453,999,253,291,210,325,234,121]
w2 = [317,252,999,202,226,108,158,140]
w3 = [165,292,201,999,344,94,124,248]
w4 = [508,210,235,346,999,336,303,94]
w5 = [222,325,116,93,340,999,182,247]
w6 = [223,235,158,125,302,185,999,206]
w7 = [410,121,141,248,93,242,199,999]



distances = {0:w0,1:w1,2:w2,3:w3,4:w4,5:w5,6:w6,7:w7}

capacity_trucks = 60
trucks = ['truck','truck']
num_trucks = len(trucks)
frontier = "---------"


def start():

    if __name__ == "__main__":

        # Constant that is an instance object 
        genetic_problem_instances =20

        print("EXECUTING ", genetic_problem_instances, " INSTANCES ")
        VRP(genetic_problem_instances)

    print(genetic_algorithm_t.x)

    y={0:(177.54,126.834),1:(173.8,126.52),2:(175.22,127.88),3:(176.4,127.17),4:(173.06,127.62),5:(176.22,127.77),6:(175.58,126.72),7:(174,127.4)}
    z1={}
    a=0
    while a<len(genetic_algorithm_t.x):
        if genetic_algorithm_t.x[a][0]=="truck":
            a+=1
        z1.update({genetic_algorithm_t.x[a][0]:y[genetic_algorithm_t.x[a][0]]})
        a+=1
    val1=list(z1.values())

    print(val1)
    print(len(val1))

    #graph stuff

    x_avrg=175.2275
    y_avrg=127.23925
    width=960
    height=540
    zoom=200
    x_adjuster=x_avrg*zoom-width/2
    y_adjuster=y_avrg*zoom-height/2
    root1=tk.Tk()
    root1.title('genetic_algorithm_t')
    root1.geometry('960x540')
    t1 =Canvas(root1,width=960,height=540,bg="white" )
    t1.pack()
    a=0
    while a<len(val1):
        t1.create_oval(val1[a][0]*zoom-x_adjuster+5,val1[a][1]*zoom-y_adjuster+5,val1[a][0]*zoom-x_adjuster-5,val1[a][1]*zoom-y_adjuster-5,fill = "blue")
        if a<len(val1)-1:
            t1.create_line(val1[a][0]*zoom-x_adjuster,val1[a][1]*zoom-y_adjuster,val1[a+1][0]*zoom-x_adjuster,val1[a+1][1]*zoom-y_adjuster,fill = "blue")
        a+=1
    root1.mainloop()



# Initialize Tkinter
root = tk.Tk()
root.title("Differential Algorithm")

# Create a Canvas widget
canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add a scrollbar to the Canvas
scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the Canvas to use the scrollbar
canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame inside the Canvas to hold the content
frame = tk.Frame(canvas)
frame.pack(pady=50)
canvas.create_window((0, 0), window=frame, anchor='n')

# Update the scroll region when the frame size changes
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", on_frame_configure)


count_button = tk.Button(frame, text="Start", command=start , width=26 )
count_button.pack(padx=5, pady=5 )

# Create a label to display status/information
count_label = tk.Label(frame, text="")
count_label.pack()

root.mainloop()