# -*- coding: utf-8 -*-
"""Decentralized Markets and the Emergence of Housing Wealth Inequality

Author: Omar A. Guerrero
Written in Pyhton 3.7
Acknowledgments: This product was developed with the support of
                The Alan Turing Institute (London).

Further details on the model and the equations can be found in the companion paper.


Example
-------



Rquired external libraries
--------------------------
- Numpy

"""

import numpy as np
import random as rd
import scipy.stats as st
from joblib import Parallel, delayed





## Agent class
class Agent:
    
    def __init__(self, delta, w, gamma, A, alpha, beta, tau, z, B, s, Lambda=1, 
                 age=18, ages=True, aging_factor=1, life_table=None, 
                 w_rand=None, alpha_rand=None, beta_rand=None, tau_rand=None, 
                 z_rand=None, B_rand=None, s_rand=None, sigmas=None):
        self.delta = delta # survival probability
        self.w = w # wage
        self.gamma = gamma # discount factor
        self.A = A # amount of common asset
        self.alpha = alpha # preference for consumption
        self.beta = beta # preference for housing
        self.tau = tau # complement of income tax
        self.z = z # complement of non-labor income tax
        self.B = B # non-labor income
        self.s = s # government transfers
        self.Lambda = Lambda # activation rate (keep it at 1)
        self.age = age # age of the agent
        self.tsA = [] # keeps track of the evolution of the asset held by the agent
        self.ages = ages # activator of the birth-death and aging processes
        self.aging_factor = aging_factor # if calendar time is relevant, this can be tuned to an explicit aging time unit
        self.life_table = life_table # data on the survival probabilities of the population
        if life_table is not None:
            # if life tables are provided, get the oldest age in the data
            self.max_age = max(list(self.life_table.keys()))
            
        # The following parameters correspond to the base levels.
        # They are used whenever an agent is reset, for example, after dying.
        # They also correspond to the mean level when parameter stochasticity is used.
        self.w0 = w
        self.gamma0 = gamma
        self.alpha0 = alpha
        self.beta0 = beta
        self.tau0 = tau
        self.z0 = z
        self.B0 = B
        self.s0 = s
        self.delta0 = delta
        self.age0 = age
        self.Lambda0 = Lambda
        self.A0 = A
        
        # If the model considers parameter stochasticity, these are standard deviations of each parameter
        self.w_rand = w_rand
        self.alpha_rand = alpha_rand
        self.beta_rand = beta_rand
        self.tau_rand = tau_rand
        self.z_rand = z_rand
        self.B_rand = B_rand
        self.s_rand = s_rand
        
        # Alternatively, the parameter standard deviations can also be provided in the form of a dictionary
        self.sigmas = sigmas
        self.vars0 = {'alpha':alpha, 'beta':beta, 'w':w, 'B':B, 'tau':tau, 'z':z, 's':s}


    def randomize(self):
        # Introduce perturbations to the parameters
        
        if self.sigmas is None:
            if self.w_rand is not None:
                rand = np.random.normal(0, self.w_rand)
                self.w = self.w0 + rand
                if self.w < 0:
                    self.w = self.w0
                    
            if self.alpha_rand is not None:
                rand = np.random.normal(0, self.alpha_rand)
                self.alpha = self.alpha0 + rand
                if self.alpha < 0 or self.alpha > 1:
                    self.alpha = self.alpha0
                
            if self.beta_rand is not None:
                rand = np.random.normal(0, self.beta_rand)
                self.beta = self.beta0 + rand
                if self.beta < 0:
                    self.beta = self.beta0
                    
            if self.tau_rand is not None:
                rand = np.random.normal(0, self.tau_rand)
                self.tau = self.tau0 + rand
                if self.tau < 0 or self.tau > 1:
                    self.tau = self.tau0
                    
            if self.z_rand is not None:
                rand = np.random.normal(0, self.z_rand)
                self.z = self.z0 + rand
                if self.z < 0 or self.z > 1:
                    self.z = self.z0
                    
            if self.B_rand is not None:
                rand = np.random.normal(0, self.B_rand)
                self.B = self.B0 + rand
                if self.B < 0:
                    self.B = self.B0
            
            if self.s_rand is not None:
                rand = np.random.normal(0, self.s_rand)
                self.s = self.s0 + rand
                if self.s < 0:
                    self.s = self.s0
        
        else:
            
            if self.age < 100:
                sigs = self.sigmas[self.age]
                for variable, sigma in sigs.items():
                    mu = self.vars0[variable]
                    if mu > 0:
                        a = 0
                        b = 2*mu
                        if variable in ['alpha', 'tau', 'z']:
                            if mu > .5:
                                diff = 1 - mu
                                a = mu - diff
                                b = 1
                        rand = np.random.normal(mu, sigma)
                        if rand >= a and rand <= b:
                            command = 'self.'+variable+' = rand'
                            exec(command)
                        else:
                            command = 'self.'+variable+' = self.'+variable+'0'
                            exec(command)

        
        

    def step(self):
        # One simulation step
        
        died = False
        self.randomize()
        if self.ages:
            self.age += self.aging_factor # add one unit to the agent's age
        self.tsA.append(self.A) # update history of held common asset
        if self.life_table is not None and self.age <= self.max_age:
            self.delta = self.life_table[self.age]
        if not self.survives(): # Bernoulli trial to decide if the agent survives
            self.age = 18 # give birth to a new agent who replaces the dead one
            died = True
        return died

    def reset(self):
        # Reset the agent's parameters to their baseline levels (used when dying)
        
        self.w = self.w0
        self.gamma = self.gamma0
        self.alpha = self.alpha0
        self.beta = self.beta0
        self.tau = self.tau0
        self.z = self.z0
        self.B = self.B0
        self.s = self.s0
        self.delta = self.delta0
        self.age = self.age0
        self.Lambda = self.Lambda0
        self.A = self.A0
        self.tsA = []


    def survives(self): 
        # Bernoulli survival trial
        if self.ages:
            return rd.random() < self.delta
        else:
            return True

    
    def compMuT(self): 
        # compute the factor that brings the utility stream to its present value (equation 2)
        return (self.gamma**self.age) / (1-self.gamma)
    
    
    def compUHat(self): 
        # compute the compactness factor frmo equation 3
        return (self.alpha**self.alpha) * ((1-self.alpha) / (self.tau*self.w))**(1 - self.alpha)


    def compUStar(self):
        # compute optimal level of utility (equation 3)
        return self.compMuT() * (1 - self.beta + self.beta*self.A) * (self.tau*self.w + self.z*self.B + self.s) * self.compUHat()


    def compBudget(self):
        # copute budget constraint
        return self.tau*self.w + self.z*self.B + self.s



    
def transactionPurchase(buyer, seller, q):
    # update the common asset held by the two parties of a transaction
    buyer.A += q
    seller.A -= q


def computeGini(array):
    # compute the Gini coefficient of a given array
    array = np.sort(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    return np.sum((2 * index - n  - 1) * array) / (n * np.sum(array))



def eqP(AB, BB, CB, MB, UB, betaB, AS, BS, CS, MS, US, betaS, h, q):
    # compute equilibrium price
    if UB==US and h==1 and betaB==betaS:
        numer = CB*MB*q*betaB+CS*MS*q*betaB
        denom = 2+AB*betaB+AS*betaB
    else:
        numer = q*(CB*MB*UB*betaB+CS*MS*US*betaS)
        denom = UB*(1+AB*betaB+q*betaB)+h*US*(1+AS*betaS-q*betaS)
    if denom != 0:
        return numer/denom
    else:
        return 0


def eqQ(AB, BB, CB, MB, UB, betaB, AS, BS, CS, MS, US, betaS, h, p):
    # compute equilibrium quantity
    if UB==US and h==1 and betaB==betaS:
        numer = 2*p+AB*p*betaB+AS*p*betaB
        denom = (CB*MB+CS*MS)*betaB
    else:
        numer = p*(UB+AB*UB*betaB+h*US*(1+AS*betaS))
        denom = CB*MB*UB*betaB-p*UB*betaB+CS*MS*US*betaS+h*p*US*betaS
    if denom != 0:
        return numer/denom
    else:
        return 0



def optQ(AB, BB, CB, MB, UB, betaB, AS, BS, CS, MS, US, betaS, h):
    # compute optimal equilibrium quantity    
    if UB==US and h==1 and betaB==betaS:
        numer = CB*MB-CS*MS+AS*CB*MB*betaB-AB*CS*MS*betaB
        denom = 2*CB*MB*betaB+2*CS*MS*betaB
        if denom != 0:
            q = numer/denom
            return q
        else:
            return 0
    elif UB*betaB == h*US*betaS:
        return 0
    else:
        const = CS*MS*betaB*betaS*(UB+h*US+AB*UB*betaB+AS*h*US*betaS)+CB*h*MB*betaB*betaS*(UB+AB*UB*betaB+h*US*(1+AS*betaS))
        root = h*(CB*h*MB+CS*MS)*betaB*betaS*(UB+h*US+AB*UB*betaB+AS*h*US*betaS)*(CB*MB*UB*betaB+CS*MS*US*betaS)*(betaB+betaS+AB*betaB*betaS+AS*betaB*betaS)
        denom = (CB*h*MB+CS*MS)*betaB*betaS*(-UB*betaB+h*US*betaS)
        if root >= 0 and denom != 0:
            q = (const - np.sqrt(root))/denom
        else:
            q = 0
        return q
    


def purchasing(buyer, seller, h):
    # given two agents, compute the optimal price and quantity to be purchased
    
    AB = buyer.A
    BB = 1 + buyer.A
    CB = buyer.compBudget()
    MB = buyer.compMuT()
    UB = buyer.compUHat()
    betaB = buyer.beta
    
    AS = seller.A
    BS = 1 + seller.A
    CS = seller.compBudget()
    MS = seller.compMuT()
    US = seller.compUHat()
    betaS = seller.beta
    
    qStar = optQ(AB, BB, CB, MB, UB, betaB, AS, BS, CS, MS, US, betaS, h)
    pStar = eqP(AB, BB, CB, MB, UB, betaB, AS, BS, CS, MS, US, betaS, h, qStar)


    if qStar <= 0 or pStar <= 0 or AS == 0:
        return (0, 0, 0)

                       
    # case 1: optimal is feasible
    if qStar <= AS and pStar <= CB:
        p = pStar
        q = qStar
        case = 1

    # case 2: price is feasible, quantity not
    elif qStar > AS and pStar <= CB:
        q = AS
        p = eqP(AB, BB, CB, MB, UB, betaB, AS, BS, CS, MS, US, betaS, h, q)
        case = 2
        
    # case 5: quantity is feasible, price not
    elif qStar <= AS and pStar > CB:
        p = CB
        q = eqQ(AB, BB, CB, MB, UB, betaB, AS, BS, CS, MS, US, betaS, h, p)
        case = 5
    
    # cases 3 and 4: neither price nor quantity are feasible
    elif qStar > AS and pStar > CB:
        qBudget = eqQ(AB, BB, CB, MB, UB, betaB, AS, BS, CS, MS, US, betaS, h, CB)
        pA = eqP(AB, BB, CB, MB, UB, betaB, AS, BS, CS, MS, US, betaS, h, AS)
        
        if qBudget <= AS:
            p = CB
            q = qBudget
            case = 4
        else:
            p = pA
            q = AS    
            case = 3
    else:
        p = 0
        q = 0
        case = 0
        
            
                
    return (p, q, case)

        

def interaction(pair, agents, h):
    
    ind_left = pair[0]
    agent_left = agents[ind_left]
    ind_right = pair[1]
    agent_right = agents[ind_right]
    
    #check that the agent is not matched with herself
    if ind_left != ind_right:
        
        buyer, seller = agent_left, agent_right       
        p, q, case = purchasing(buyer, seller, h)
        
        if case == 0:
            buyer, seller = agent_right, agent_left   
            p, q, case = purchasing(buyer, seller, h)
            
        # check that buyer has budget and that seller has asset
        if case != 0 and q <= seller.A and q > 0 and p > 0 and p <= buyer.compBudget():
            
            # make transaction happen: update allocations and outcome histories
            transactionPurchase(buyer, seller, q)
            
            return (p, q, case)




def pair_util(pair, agents, h):
    # Compute the surplus utility that a pair of agents would get if theu transact
    
    ind_left = pair[0]
    agent_left = agents[ind_left]
    ind_right = pair[1]
    agent_right = agents[ind_right]
    
    #check that the agent is not matched with herself
    if ind_left != ind_right:
        
        buyer, seller = agent_left, agent_right       
        p, q, case = purchasing(buyer, seller, h)
        buyer_id = ind_left
        seller_id = ind_right
        
        if case == 0:
            buyer, seller = agent_right, agent_left   
            p, q, case = purchasing(buyer, seller, h)
            buyer_id = ind_right
            seller_id = ind_left
            
        # check that buyer has budget and that seller has asset
        if case != 0 and q <= seller.A and q > 0 and p > 0 and p <= buyer.compBudget():
            
            #get utility surplus from buyer in equilibrium (equal to the seller's)
            AB = buyer.A
            CB = buyer.compBudget()
            MB = buyer.compMuT()
            UB = buyer.compUHat()
            betaB = buyer.beta
            
            utility_surplus = -UB*(p+AB*p*betaB-CB*MB*q*betaB+p*q*betaB)
        
        else:
            
            utility_surplus = 0
            
        return (utility_surplus, buyer_id, seller_id, case)




def match(agents, h=1, it=0, encounters=1):
    # Perform a multi-round matching procedure and return a list with all pair configurations.
    # When encounters=1 there is only one round, and this corresponds to the version used in the paper.
    
    N = len(agents)
    inds = list(range(N))
    N2 = int(N/2)
    
    Nsamp = int(N)
    Nsamp -= Nsamp%2
    N2 = int(Nsamp/2)
    
    tuples = []
    
    for round in range(encounters):
        inds_rand = np.array(rd.sample(inds, Nsamp))
        inds_left = inds_rand[0 : N2]
        inds_right = inds_rand[N2 : Nsamp]
        pairs = list(zip(inds_left, inds_right))
    
        for pair in pairs:
            output = pair_util(pair, agents, h)
            tuples.append(output)
        
    tuples = sorted(tuples, key=lambda x: x[0])[::-1]
    
    final_pairs = []
    checked_agents = []
    for tup in tuples:
        utility, buyer, seller, case = tup
        if utility > 0 and case != 0:
            if buyer not in checked_agents and seller not in checked_agents:
                final_pairs.append((buyer, seller))
                checked_agents.append(buyer)
                checked_agents.append(seller)
                
    return final_pairs
        
        


def model_purchase(agents, h=1, it=0, encounters=1, maxIter=100, n_jobs=None, verbose=True):
    # run the model for a number of periods (100 by default)
    # needs a population of agents already instantiated (it should be an even number)
    # needs a parameter h for the complement of the sales tax (1-h is the tax)
    
    N = len(agents)
    
    # arrays that will store aggregate variables
    ts_gini = []
    ts_cases = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    ts_p = []
    ts_q = []
    ts_p_std = []
    ts_q_std = []
    
    # taxes collected by the state to be redistributed every period
    sales_tax_pot = 0
    inherit_tax_pot = 0
    
    # iterate
    for itera in range(maxIter):
        if verbose:
            print(itera, end=' ')
            
        # # split the population in two and randomly match them
        # inds_rand = np.array(rd.sample(inds, Nsamp))
        # inds_left = inds_rand[0 : N2]
        # inds_right = inds_rand[N2 : Nsamp]
        # pairs = list(zip(inds_left, inds_right))
        
        # arrays to store all outcomes of this iteration
        cases = []
        prices = []
        quantities = []
        
        # advance the agent's life in one step
        for agent in agents:
            died = agent.step()
            if it > 0 and died:
                inherit_tax_pot += agent.A*it
                agent.A = agent.A*(1-it)
            
        # distribute taxes
        if h < 1 or it > 0:
            for agent in agents:
                agent.s += sales_tax_pot/N
                agent.A += inherit_tax_pot/N
            sales_tax_pot = 0
            inherit_tax_pot = 0
        
        # perform multiple rounds matches and get final pairs
        pairs = match(agents, h=h, it=it, encounters=encounters)
        
        # go through each matched pair
        if n_jobs is None:
            for pair in pairs:
                output = interaction(pair, agents, h)
                
                if output is not None:
                    p, q, case = output            
                    prices.append(p)
                    quantities.append(q)
                    cases.append(case)
                    sales_tax_pot += p*(1-h)
                    
        else: # parallel computation
            results = list(Parallel(n_jobs=n_jobs, verbose=0)(delayed(interaction)(pair, agents, h) for pair in pairs))
            all_outputs = [output for output in results if output is not None]
            for output in all_outputs:
                p, q, case = output            
                prices.append(p)
                quantities.append(q)
                cases.append(case)
                sales_tax_pot += p*(1-h)
            
        # compute Gini coefficient
        gini = computeGini([a.A for a in agents])
        ts_gini.append(gini)
        
        # if there were transactions, compute the average price
        if len(quantities) > 0:
            ts_p.append(np.mean(sum(prices)/sum(quantities)))
        else:
            ts_p.append(0)
            
        # update histories of aggregate variables
        ts_q.append(np.mean(quantities))
        ts_p_std.append(np.std(prices))
        ts_q_std.append(np.std(quantities))
        
        # update counts of transaction cases
        for case in range(1,7):
            ts_cases[case].append(np.sum(np.array(cases)==case))
            
        # reset government transfers
        if h < 1:
            for agent in agents:
                agent.s = agent.s0
                
    
    print(' ')
    return ts_gini, ts_cases, ts_p, ts_q, ts_p_std, ts_q_std










    





    





    


    