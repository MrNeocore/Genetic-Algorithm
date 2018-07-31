####### PART 1.A - EA ####### 
# Name          :   MEYER Jonathan
# Student ID    :   HW00281038
# Date          :   Oct. 1st 2017
##############################

import random
import math
import numpy as np
import itertools
import copy
import time 
import pandas as pd
import matplotlib.pyplot as plt
import profile
import functools
import operator
import time
from random import shuffle
import heapq
from statistics import mean
from operator import methodcaller

# TSP_Cost function implemented in Cython. Note : Would have to recompile in order to rename...
import test_fast 

def checkData(data):
    return True
    if len(data) != 29:
        return False
        
    if len(data) > len(set(data)):
        return False
        
    return True

def checkCityDistances():
    trav = Traveler(range(0,Traveler.encoding['lenght']))
    del(trav.data[0])
    #trav.data.append(trav.data[0]) 
    
    #for x in range(1,Traveler.encoding['lenght']-1): 
    #    distance = test_fast.TSP_Cost(Traveler.encoding['dataset'][trav.data[x]][1], Traveler.encoding['dataset'][trav.data[x]][2], Traveler.encoding['dataset'][trav.data[x+1]][1], Traveler.encoding['dataset'][trav.data[x+1]][2])
    #    print(f"Distance between city {x} and city {x+1} : {distance}") 
    
    geoPlot(trav)

def fitnessPlotFromFile():
     data = [line.strip() for line in open("logs/last_fitness_record", 'r')][1:] # If non existant ?
     lst = []
     for x in data:
        lst.append(x.split(';'))
        lst[-1] = list(map(int,lst[-1])) # Convert strings to int
     
     fitnessPlot(lst, 0, True)
     
def fitnessPlot(fitness, last, new_figure = False): # Part of this should be moved to the init phase so that it is not executed multiple times unnecessarily
    if new_figure:
        plt.figure(500)
    else:
        plt.figure(300)
        
    plt.clf()
    gen = [x[0] for x in fitness[-last:]]
    fit = [x[1] for x in fitness[-last:]]

    plt.plot(gen, fit)
    plt.xlabel('Generation count')
    plt.ylabel('Best individual fitness')
    plt.title('Fitness vs generations')  
    #plt.text(gen[0]+10, fit[0], f'Current fitness : {fit[-1]}')
    plt.legend()
    plt.draw()
    plt.pause(0.01)    
    
def geoPlot(best):
    plt.figure(200)
    best.data.append(best.data[0])
    DATA = Traveler.encoding['dataset']

    for idx in range(len(best.data)-1):
        plt.plot((DATA[best.data[idx]][2],DATA[best.data[idx+1]][2]),(DATA[best.data[idx]][1],DATA[best.data[idx+1]][1]), marker = 'o')
   
    plt.draw()
    plt.pause(0.001)
    
    
class GA:
    stall_options = {'abort': 1, 'rm-dup':2, 'rm-dup-bts':3, 'ignore':4}
    not_init = True                  
                    
    def __init__(self, config):
        self.settings = config.settings
        self.settings['encoding']['lenght'] = len(self.settings['encoding']['dataset'])-1
        self.settings['encoding']['span'] = list(range(1,len(self.settings['encoding']['dataset'])))
        self.pop_size = self.settings['pop']['pop_size']    # Shorter alias 
        Traveler.setSettings(self.settings)
        self.init_pop()
        self.fitness_record = []
        
    def init_pop(self):
        self.population = []
                
        # Create a 10*$(pop_size) population 
        for x in range(0,self.pop_size*self.settings['pop']['init_factor']):
            self.population.append(Traveler())
            
        # Keep the best ones
        self.sortPopulation()
        self.population = self.population[:self.pop_size]
    
    def crossover(self, parents_ids):   
        algo_name = self.settings['algo']['crossover'] 
        
        #print(f"Using crossover {algo_name}")
        if algo_name == 'one-point-co':
            for x in pop:
                cross_indiv = self.population[random.randrange(0,self.pop_size)]
                x.crossover(cross_indiv)
        
        
        elif algo_name == 'pmx':
            p_fit = []
            p_fit.append(self.population[parents_ids[0]].getFitness())
            p_fit.append(self.population[parents_ids[1]].getFitness())
            
            x1_t = random.randrange(0,self.settings['encoding']['lenght'])
            x2_t = random.randrange(0,self.settings['encoding']['lenght'])
            
            x1 = min([x1_t,x2_t])   # x1 > x2 otherwise list slices don't work
            x2 = max([x1_t,x2_t])
            
            chunk1 = self.population[parents_ids[0]].data[x1:x2+1]
            chunk2 = self.population[parents_ids[1]].data[x1:x2+1]
            
            coor1 = {}
            coor2 = {}
            for idx, x in enumerate(chunk1):
                coor1[x] = chunk2[idx]
                
            for idx, x in enumerate(chunk2):
                coor2[x] = chunk1[idx]
            
            
            child1_data = [None] * self.settings['encoding']['lenght']
            child2_data = [None] * self.settings['encoding']['lenght']
            
            child1_data[x1:x2+1] = chunk2[:]
            child2_data[x1:x2+1] = chunk1[:]
            
            for idx in range(0, self.settings['encoding']['lenght']):
                if idx < x1 or idx > x2:
                    p1_val = self.population[parents_ids[0]].data[idx]
                    if p1_val not in coor2:
                        child1_data[idx] = p1_val
                    else:
                        while p1_val in coor2:
                            p1_val = coor2[p1_val]
                        child1_data[idx] = p1_val
                        
            for idx in range(0, self.settings['encoding']['lenght']):
                if idx < x1 or idx > x2:
                    p2_val = self.population[parents_ids[1]].data[idx]
                    if p2_val not in coor1:
                        child2_data[idx] = p2_val
                    else:
                        while p2_val in coor1:
                            p2_val = coor1[p2_val]     
                        child2_data[idx] = p2_val
            
            assert(checkData(child2_data))
            assert(checkData(child1_data))

            children_arr = []
            children_arr.append(Traveler(child1_data))
            children_arr.append(Traveler(child2_data))
          
            return children_arr
    
    def select(self, nb, override_algo = None):
        if override_algo == None:
            select_algo = self.settings['algo']['select']
        else:
            select_algo = override_algo
        
        ret_pop = []
        for _ in range(0,nb):
            if select_algo[0] == 'bts':
                #print(f"Using select {select_algo}")
                # Tournament population
                trm_ids = random.sample(range(0, len(self.population)), int(select_algo[1] * len(self.population) / 100)) # Can't use pop size if using elitsm, len(pop) != pop_size for now
                
                best_id = trm_ids[0]
                best_fitness = self.population[best_id].getFitness()
                
                # Get best individual from tournament
                for idx in trm_ids:
                    fitness = self.population[idx].getFitness()            # Avoid recalculating fitness everytime
                    if fitness < best_fitness:
                        best_id = idx
                        best_fitness = fitness
                
                # Append selected individual to the list
                ret_pop.append(best_id)
         
        return ret_pop
    
    def roulette(self, nb, individuals): 
    # roulette with high biais
    
        if(nb >= len(individuals)):
            raise Exception("Roulette must have more input individuals than output individuals : nb < len(individuals)")
        if(nb == 0 or len(individuals) <= 1):
            raise Exception("Roulette input count must be greater than 1 - output must be greater than 0")
            
        indiv_fitness = []
        for indiv in individuals:
            indiv_fitness.append(indiv.getFitness())
        
        # Product much faster than exponentiation. 6-7x
        sum_fitness = sum(indiv_fitness)
        real_fitness = [(sum_fitness-x)*(sum_fitness-x) for x in indiv_fitness]
        indiv_fitness_norm = [x/sum(real_fitness) for x in real_fitness]
        
        assert(round(sum(indiv_fitness_norm), 9) == 1.0) # Level to which numpy doesn't complain if sum != 1.0. Ex : p=[0.01,0.98999999] is fine
        
        idx = []
        for n in range(nb):
            new_id = np.random.choice(range(len(individuals)), p=indiv_fitness_norm)
            while new_id in idx:    # Not optimized... 
                new_id = np.random.choice(range(len(individuals)), p=indiv_fitness_norm)
            idx.append(new_id)
        
        return [individuals[id_] for id_ in idx] 
        
        
    def nextGeneration(self):
        update_algo = self.settings['algo']['update']
        co_algo = self.settings['algo']['crossover']
                  
        if update_algo[0] == 'elitism':
            self.sortPopulation()
            # Current best individuals
            kept_index = math.floor(update_algo[1] * self.pop_size / 100)
            
            # Keep only the best ones ! 
            old_pop = self.population
            self.population = self.population[:kept_index]
            
            if co_algo != None:
                # Replenish population with children coming from crossover + mutated
                for _ in range(0,int((self.pop_size - kept_index)/2)):
                        children = self.crossover(self.select(2)) 
                        for child in children:
                            self.population.append(child)
                assert(self.population != old_pop)
                
                
            # Truncation algorithm        
            else:
                # Replenish population with mutated copies of the best ones
                while(len(self.population) != self.pop_size):
                    best = self.population  # Temporary variable, can't append to the list being iterated over
                    for x in best:
                        new_indiv = Traveler(x.data)
                        new_indiv.mutate()
                        self.population.append(new_indiv)
                
        else:
            # Update rounds
            for _ in range(0, int(update_algo[1] * self.pop_size / 100)):
            
                # Select
                parents_ids = self.select(2)
                p_fit = []
                p_fit.append(self.population[parents_ids[0]].getFitness())
                p_fit.append(self.population[parents_ids[1]].getFitness())
                
                # Crossover
                if co_algo != None:
                    children = self.crossover(parents_ids)
                    assert(len(children) == 2)
                    assert(checkData(children[0].data))
                    assert(checkData(children[1].data))
                    assert(self.population[parents_ids[0]].getFitness() == p_fit[0])
                    assert(self.population[parents_ids[1]].getFitness() == p_fit[1])
                else:
                    children = [Traveler(self.population[x].data) for x in parents_ids]
                
                # Mutate
                for x in children:
                    x.mutate()
                    
                # So that we replace optimally. Ex : p1 = 3, p2 = 7 must be replaced by ch1 = 9, ch2 = 5 in this order -> result : 7,9, otherwise 5,9
                children.sort(key=methodcaller('getFitness'), reverse=not self.settings['encoding']['maximize'])

                if self.population[parents_ids[0]].getFitness() > self.population[parents_ids[1]].getFitness():
                    parents_ids[0], parents_ids[1] = parents_ids[1], parents_ids[0]
                
                if update_algo[0] == 'proba-replace-parent':
                    indiv = children
                    indiv.extend([self.population[id_] for id_ in parents_ids])
                    replacement = self.roulette(2,indiv)
                    for idx in range(len(replacement)):
                        self.population[parents_ids[idx]] = replacement[idx]
                        
                     
                if update_algo[0] == 'replace-parent':
                    # Replace (parents)
                    for idx in range(0, 2):
                        ch_fit = children[idx].getFitness()
                        if ch_fit < p_fit[idx]:
                            self.population[parents_ids[idx]] = children[idx]
                            assert(ch_fit < p_fit[0] or ch_fit < p_fit[1])
                            
                            
                elif update_algo[0] == 'replace-worst':
                    #print(f"Using update {update_algo}")
                    self.sortPopulation()
                    for idx in range(0, 2):
                        ch_fit = children[idx].getFitness()
                        worst_fit = self.population[-2+idx].getFitness()    # -2 + 0 = -2 : 2sd worst, replaced by best children, -2 + 1 = -1 : worst, replaced by worst child
                        if ch_fit < worst_fit:
                            self.population[-2+idx] = children[idx]
                         
        # Used to check for any "population contamination" - ie. the data field of 2 individuals are pointing at the same memory space - they are linked -> reduced diversity.
        #for x in range(0, self.pop_size):
        #    for y in range(0,self.pop_size):
        #        if x != y:
        #            assert(self.population[x] is not self.population[y])
        #            assert(self.population[x].data is not self.population[y].data)
                
                
    # Used to re-fill the population. Necessary when removing duplicates or using 'elitism' update scheme
    def fill(self):
        while(len(self.population) < self.pop_size):
            self.population.append(Traveler())
    
    
    def sortPopulation(self):
        self.population.sort(key=methodcaller('getFitness'), reverse=self.settings['encoding']['maximize'])
        
        
    def getPopFitness(self, size=0):
        if size == 0:
            size = self.pop_size
            
        return [x.getFitness() for x in self.population[0:size]]
     
    # Returns a string containing information about the current generation population 
    def getPop(self, size = 0, pop_list = None):        
        if pop_list == None:
            pop_list = self.population
            
        if size == 0:
            size = len(pop_list)
            
        text = [str(x.id) + " - Fitness : " + str(x.getFitness()) for x in pop_list[:size]]
        string = '\n'.join(str(x) for x in text)
        return "Generation : {}\n".format(self.gen_count) + str(string) + "\nTraveler created count : {}".format(Traveler.created_count) + "\n"
        
    # Starts the GA
    def start(self):
        self.gen_count = 0
        
        # Varibles used to stop the GA on specific goals
        max_gen = self.settings['stop']['max_gen']
        max_time = self.settings['stop']['max_time']
        min_perf = self.settings['stop']['aim']
        output = self.settings['output']
        stop_on_perf = (min_perf != 0)
        stop_on_time = (max_time != 0)
        stop_on_gen = (max_gen != 0)
        perf_stop = False
        time_stop = False
        gen_stop = False
        
        # Determines how often the output is made - every X generations
        calc_interval = output['out_interval']
        
        # Used to detect that the GA is stuck in a local minima
        datacheck_interval = 2*calc_interval
        top_count = int(self.pop_size/50) #Top 2%
        previous_top = []
        
        last_calc = 0
        last_check = 0
        
        start_time = time.time()
        
        # Prevents multiple or unnecessary matplotlib plot initialization
        if output['mode'] == 'plot' or output['mode'] == 'full' and GA.not_init == True:
            plt.ion()
            GA.not_init = False
        
        # Main GA loop
        while time_stop == False and gen_stop == False and perf_stop == False:
            self.nextGeneration()
            
            # Whether or not it's time to display information to the user - and check if any goal is reached
            if last_calc > calc_interval:
                self.sortPopulation()
                pop_fitness = self.getPopFitness(5)
                self.fitness_record.append((self.gen_count, self.population[0].getFitness()))
                
                # Goals check
                if stop_on_perf and pop_fitness[0] >= min_perf:
                    perf_stop = True
                
                if stop_on_time and time_elapsed > max_time:
                    time_stop = True
                    
                if stop_on_gen and self.gen_count > max_gen:
                    gen_stop = True
                
                # User output
                if any(x in ['text','full'] for x in output['mode']):
                    print(self.getPop(output['perf_ref']))
                    
                # Displays a "map" of the cities - useless except if the GA is actually working well...
                if any(x in ['geoplot','plot','full'] for x in output['mode']):
                    geoPlot(self.population[0])
                 
                # Displays partial fitness/generation curve
                if any(x in ['fitplot','plot','full'] for x in output['mode']):
                    fitnessPlot(self.fitness_record, 50)
                    
                last_calc = 0
                
            else:
                last_calc +=1
            
            
            # Local minima detection
            if last_check >= datacheck_interval:
                new_top = [x.getFitness() for x in self.population[:top_count]]
                
                # Stall detected
                if new_top == previous_top :
                    if output['stall_action'] == 'manual':
                        print("Suspected local minimal detected - what do you want to do :")
                        print("1. Abort")
                        print("2. Remove all duplicates")
                        print("3. Remove duplicates and apply bts to the remainder")
                        print("4. Ignore")
                        choice = int(input())
                    
                    else:
                        choice = GA.stall_options[output['stall_action']]
                    
                    if choice == 1:
                        gen_stop = True
                    elif choice == 2:
                        self.cleanPop('rm-duplicates')
                        self.sortPopulation()
                        new_top = [x.getFitness() for x in self.population[:top_count]]
                    elif choice == 3:
                        self.cleanPop('rm-duplicates', 'bts')
                        self.sortPopulation()
                        new_top = [x.getFitness() for x in self.population[:top_count]]
                                
                previous_top = new_top
                last_check = 0  
                
            else:
                last_check +=1     

            
            self.gen_count +=1
            time_elapsed = time.time() - start_time
            
        # Shows plots when GA is done and records the fitness/generation data to a file for further plotting
        if output['mode'] != 'none':
            geoPlot(self.population[0])
            fitnessPlot(self.fitness_record, 0)
            
            with open('logs/last_fitness_record', 'w') as f:
                f.write("generation;fitness\n")
                for x in self.fitness_record:
                    f.write(f"{x[0]};{x[1]}\n")
           
            # Shows the full fitness/generation curve
            fitnessPlotFromFile()
        

        if perf_stop == True:
            return ("Desired fitness reached !  -  {} generations and {} seconds".format(self.gen_count, time_elapsed), (self.gen_count, time_elapsed, self.getPopFitness(1)[0]))   
        elif time_stop == True:
            return ("Excedeed max time !  -  {} generations and {} seconds".format(self.gen_count, time_elapsed), (self.gen_count, time_elapsed, self.getPopFitness(1)[0]))
        elif gen_stop == True:
            return ("Excedeed max generation count !  -  {} generations and {} seconds".format(self.gen_count, time_elapsed), (self.gen_count, time_elapsed, self.getPopFitness(1)[0]))

    # Used when stuck in a local minima - removes duplicated and apply bts to the remaining population
    def cleanPop(self, param, param2 = ''):
        
        if param == 'rm-duplicates':
            print("Removing duplicates from population - please wait...")
            new_pop = []       
            
            for indiv in self.deduplicatePop():
                new_pop.append(indiv)
            
            # Non duplicated population
            self.population = new_pop       
        
        if param2 == 'bts':   
            print("Applying BTS to the deduplicated population - please wait...")
            
            # Keep half the remaining population
            self.population = self.select(int(len(new_pop)/2), override_algo = 'bts')  
            
            print("Removing duplicates from bts population - please wait...")
            # Re-deduplicate
            for indiv in self.deduplicatePop():
                new_pop.append(indiv)
            
        print("Replacing missing individuals by new random - please wait...")    
        self.fill() 
         
    def deduplicatePop(self):
        seen = set()
        for indiv in self.population:
            fit = indiv.getFitness()
            if not fit in seen:
                seen.add(fit)
                yield indiv    

# GA individual class - TSP cities visiting order
class Traveler:
    newid = itertools.count()
    created_count = 0
    
    def __init__(self, data = ""):
        Traveler.created_count += 1
        self.id = next(Traveler.newid)
        self.mut_count = 0
        self.has_mut = True
        
        if data == "": 
            self.data = list(Traveler.encoding_data)
            shuffle(self.data)
     
        else:
            self.data = list(data)            
              
    def setData(self, data):
        self.data = data  
        
    @classmethod
    def setSettings(cls, problem_settings):
        Traveler.encoding = problem_settings['encoding']
        Traveler.mutation = problem_settings['algo']['mutate']
        Traveler.cross_over = problem_settings['algo']['crossover']
        Traveler.encoding_data = [_ for _ in range(min(Traveler.encoding['span']), max(Traveler.encoding['span'])+1)]
       
    def getFitness(self):
        if(self.has_mut):
            total_len = 0
            self.data.append(self.data[0]) # Go back to first city

            for x in range(0,Traveler.encoding['lenght']-1): 
                total_len += test_fast.TSP_Cost(Traveler.encoding['dataset'][self.data[x]][1], Traveler.encoding['dataset'][self.data[x]][2], Traveler.encoding['dataset'][self.data[x+1]][1], Traveler.encoding['dataset'][self.data[x+1]][2]) 

            del self.data[-1]
            
            self.fitness = total_len
            self.has_mut = False
            return total_len
            
        else:
            return self.fitness             
              
    def mutate(self):
        self.mut_count += 1
        self.has_mut = True
       
        # M-distinct-gene new-allele mutation (random or normal distribution) - not relevant with TSP
        if type(Traveler.mutation[0]) == tuple and Traveler.mutation[0][0] == 'n-random':
            for _ in range(0, self.mutation[0][1]):
                rand_gene = random.randrange(0, len(self.data))
                self.data[rand_gene] = self.getGeneVal(self.data[rand_gene])
              
        # Genewise mutation - not relevant with TSP
        elif Traveler.mutation[0] == 'genewise':
            for x in range(0,Traveler.encoding['lenght']):
                if random.choice('0000000000000000000000001'):          # Better but slower : np.random.choice(2,1,p=[24/25,1/25])   10-20x slow 
                    self.data[x] = self.getGeneVal(self.data[x])  
        
        # Adjacent-swap
        elif Traveler.mutation[0] == 'adj-swap':
            rand_pos = random.randrange(1,Traveler.encoding['lenght'])
            self.data[rand_pos-1], self.data[rand_pos] = self.data[rand_pos], self.data[rand_pos-1]  
                                        
        # Exchange mutation : random-swap               
        elif Traveler.mutation[0] == 'em':
            rand_pos1 = random.randrange(0,Traveler.encoding['lenght'])
            rand_pos2 = random.randrange(0,Traveler.encoding['lenght'])
            self.data[rand_pos1], self.data[rand_pos2] = self.data[rand_pos2], self.data[rand_pos1]
        
        
        # Inversion mutation : (1[23]4) -> (14[32])
        elif Traveler.mutation[0] == 'ivm':
            lenght = len(self.data)
            x1_t = random.randrange(0,Traveler.encoding['lenght'])
            x2_t = random.randrange(0,Traveler.encoding['lenght'])
            x1 = min([x1_t,x2_t])   # x1 > x2 otherwise list slices don't work
            x2 = max([x1_t,x2_t])
            
            # Save and reverse chunk
            chunk = self.data[x1:x2+1]
            chunk = chunk[::-1] # Reverse chunk
            
            count = 0
            # Remove chunk
            for _ in range(x1,x2+1):
                count += 1
                del self.data[x1]           # Removing displaces elements... remove x time [x1] removes [x1..x1+x]
            
            assert(count == len(chunk))
            insert_pt = random.randrange(0,Traveler.encoding['lenght']-len(chunk)+1)
            
            for x in range(0, len(chunk)):
                self.data.insert(insert_pt+x, chunk[x])
            assert(len(self.data) == lenght)
            
        # Simple inversion mutation - Note : Wrongly typed... should be SIM - kept for consistency
        elif Traveler.mutation[0] == 'ism':
            start_data = self.data
            x1_t = random.randrange(0,int(Traveler.encoding['lenght']/5))  ##### VARIBLE
            x2_t = random.randrange(0,int(Traveler.encoding['lenght']/5))
            x1 = min([x1_t,x2_t])   # x1 > x2 otherwise list slices don't work
            x2 = max([x1_t,x2_t])
            
            # Save and reverse chunk
            chunk = self.data[x1:x2+1]
            chunk = chunk[::-1] # Reverse chunk
            self.data[x1:x2+1] = chunk
            
    # Used to get a new random value - following a normal or uniform distribution
    @classmethod
    def getGeneVal(self, prev_val):

        if Traveler.mutation[1] == 'random':
            return random.randrange(min(Traveler.encoding['span'],max(Traveler.encoding['span'])))
                       
        elif Traveler.mutation[1] == 'normal':
            new_value = abs(np.random.normal(prev_val, Traveler.mutation[2], 1)[0])     # Reverb value < min_encoding
            max_encoding = max(Traveler.encoding['span'])
            min_encoding = min(Traveler.encoding['span'])
            new_value = int(round(new_value))
            if new_value > max_encoding:
                new_value = max_encoding - (new_value - max_encoding)                   # Reverb value > max_encoding
            if new_value == 0:
                new_value = min_encoding
            return new_value
        
        assert(work)

# Used to configure the GA algorithm - not necessary per se - but why not.
class GA_configurator:    
    # Only these valid_settings will be kept
    valid_settings={'pop': ['pop_size', 'init_factor'], 'algo': ['mutate', 'select', 'update', 'crossover'], 'stop': ['max_gen', 'max_time', 'aim'], 'output': ['mode', 'perf_ref', 'stall_action', 'out_interval'], 'encoding': ['dataset', 'maximize']}
    
    def __init__(self):
        self.settings = {}
        
    def conf(self, **kwargs):
        if 'setting' in kwargs:                                                     # Is the setting type provided
            setting_type = kwargs['setting']
            if setting_type in GA_configurator.valid_settings:                      # Is the setting type valid
                self.settings[setting_type] = {}
                for param in kwargs:                                                # Is the setting option valid
                    if param in GA_configurator.valid_settings[setting_type]:
                        self.settings[setting_type][param] = kwargs[param]          # Recording setting in the appropriate dictionnary section
    
    
# Used to benchmark a list of algorithm against each other - aka "the time eater".
class GA_benchmark:
    settings_list = {'pop':[1000],
                    'mutate':[('ivm', 'ism')],
                    'select':[('bts',15), ('bts',40)],
                    'update':[('replace-worst', 15), ('replace-parent',15), ('proba-replace-parent',15)],
                    'crossover' : ['pmx', None] }
                    
    def start(self):
        conf = GA_configurator()
        
        conf.conf(setting = 'stop', max_gen = 0, max_time = 300, aim = 0)
        conf.conf(setting = 'output', mode = 'none', perf_ref = 10, out_interval = 50, stall_action = 'ignore')
        
        bench = []
        run = 0
        file_path = "logs/TSP_{}_benchmark.txt".format(time.strftime("%d-%m-%y_%H:%M:%S"))
        
        dataset = loadDataSet("data/Luxembourg_opti.txt")
        
        with open(file_path, 'w') as log:
            log.write("pop_size;mutate;select;crossover;update;gen_count;fitness\n")
        
            for pop_set in GA_benchmark.settings_list['pop']:
                for mut_set in GA_benchmark.settings_list['mutate']:
                    for sel_set in GA_benchmark.settings_list['select']:
                        for co_set in GA_benchmark.settings_list['crossover']:
                            for up_set in GA_benchmark.settings_list['update']:
                                conf.conf(setting = 'pop', pop_size = pop_set, init_factor = 3)
                                conf.conf(setting = 'algo', mutate = mut_set, select = sel_set, update = up_set, crossover = co_set)
                                conf.conf(setting = 'encoding', dataset = dataset, maximize = False)
                                print("Testing {} {} {} {} {}".format(pop_set, mut_set, sel_set, up_set, co_set))
                                
                                bench.append(run)
                                bench[run] = {}
                                bench[run]['settings'] = []
                                bench[run]['results'] = []
                                bench[run]['settings'].append((pop_set, mut_set, sel_set, co_set))
                                
                                print("Now running run {} / {}".format(run+1, functools.reduce(operator.mul,(len(x[1]) for x in GA_benchmark.settings_list.items()),1)))
                                
                                # Run multiple times to average out - use 1 for maximum speed...
                                for subrun in range(0,1):
                                    a = GA(conf)
                                    result = a.start()[1]
                                    
                                    bench[run]['results'].append(result)
                                    print(".", end = "", flush = True)
                                    
                                   
                                print(bench[run]['results'])
                                
                                print("")

                                log.write("{};{};{};{};{};{};{}\n".format(pop_set,
                                                                       mut_set, 
                                                                       sel_set,
                                                                       co_set,
                                                                       up_set,
                                                                       mean([x[0] for x in bench[run]['results']]), 
                                                                       mean([x[2] for x in bench[run]['results']])))
                                run += 1                        
          
 
# Plots a benchmark result - can be restricted to the X best combination
def plot(file_name,x_data,y_data, keep_best_nb = 0, print_best = False):
    
    # Activate interactive mode for non-blocking plotting
    plt.ion()
    data = pd.read_csv(file_name, sep=";")
    if keep_best_nb:
        data = data.nsmallest(keep_best_nb, y_data)
        
    if x_data != 'all':
        xy_data = data[[x_data,y_data]]
        xy_data = xy_data.groupby([x_data], as_index=False).mean()
        xy_plot = xy_data.plot(kind='bar', x=x_data)
        for tick in xy_plot.get_xticklabels():
            tick.set_rotation(0)
    
    # Plots the whole run configuration against the fitness      
    else:       
        x_data = 'algorithms'
        y_data = 'fitness'
        data['algorithms'] = data['pop_size'].astype(str) + ";" + data['mutate'].astype(str) + ";" + data['select'].astype(str) + ";" + data['crossover'].astype(str) + ";" +  data['update'].astype(str)
        xy_data = data[[x_data, y_data]]
        xy_data = xy_data.groupby([x_data], as_index=False).mean()
        xy_plot = xy_data.plot(kind='bar', x=x_data)  
    
    if print_best:
        for x,y in zip(data[x_data], data[y_data]):
            print(f"{x} : {y}")    
    plt.show()

# Load a optimized datased (as outputed by the "TSP_data_preprocessing.py" script)
def loadDataSet(file_path):
    data = []
    with open(file_path, "r") as ds:
        data.append("EMPTY") # So that city id are aligned with list indexes
        for i, line in enumerate(ds):
            #if line.startswith('1') or i > 6:
            line = line.strip().split(" ")
            data.append((int(line[0]), float(line[1]), float(line[2])))
    return data                   
      
 
 
if __name__ == "__main__":
    # GA Configuration
    conf = GA_configurator()
    conf.conf(setting = 'pop', pop_size = 500, init_factor = 2)
    conf.conf(setting = 'algo', mutate = ('ivm',), select = ('bts', 15), crossover = 'pmx', update = ('replace-worst',15))
    conf.conf(setting = 'stop', max_gen = 0, max_time = 120, aim = 0)
    conf.conf(setting = 'encoding', dataset = loadDataSet("data/Luxembourg_opti.txt"), maximize = False)
    conf.conf(setting = 'output', mode = ['text', 'fitplot'], perf_ref = 10, out_interval = 25, stall_action = 'manual')

    # Standard GA mode - slow to show the first results - please wait...
    a = GA(conf)
    print(a.start()[0])

    # Profiler mode - SLOW
    #profile.run('a.start()[0]; print()')
    
    # Benchmark mode
    #bench = GA_benchmark()
    #bench.start()

    ##### PLOTTING ####
    # Print the last fitness/generation curve file - data file provided
    #fitnessPlotFromFile()
    
    # You can run that, the data file are provided in the archive
    #plot("logs/24_5min.txt", 'all', 'fitness',keep_best_nb=4, print_best=True)
    #plot("logs/24_5min.txt", 'pop_size', 'fitness')
    #plot("logs/24_5min.txt", 'mutate', 'fitness')
    #plot("logs/24_5min.txt", 'select', 'fitness')
    #plot("logs/24_5min.txt", 'crossover', 'fitness')
    #plot("logs/24_5min.txt", 'update', 'fitness')

    # Used to check city distances consistency (visual vs number)
    #checkCityDistances()
   
    # Wait for 'enter' key press so that we can interact with the graphs before exiting
    input("Press enter to exit\n")
    