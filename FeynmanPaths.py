from random import randint
import numpy as np
import vegas

#INITIAL VALUES
x_0 = 0
tau = 4
x = 0
N = 7
m = 1
k = 100 #number of paths generated
bound = 2

def V(x):
    return 0.5*(x**2)



'''

#Random Path Generator: set t_0 = 0

def randompathgen(x_0, tau, x, N):

    time = (tau/N)*np.arange(N+1)  #array for timepoints, includes t_0 = 0 and tau
    x_j = np.random.default_rng()  #random number generator for x_j values, faster than random module for large arrays - better statistical properties and no hidden global state
    path = np.empty(N+1)
    path[0] = x_0
    path[-1] = x
    path[1:-1] = x_j.uniform(-10000,10000,N-1)

    return np.column_stack((time, path))


#print(randompathgen(0,10,1000,15))



#Summed Energy Along Path:


def summed_energy(m, paths, tau, N, V):     #V is potential and is a function of position/the midpoint between x_j and x_(j+1)

    v_avg = (paths[1:] - paths[:-1])/(tau/N)
    midpoint = 0.5 * (paths[1:] + paths[:-1])

    return np.sum((0.5*m*(v_avg**2))+ V(midpoint))



#array of several paths

def all_paths(x_0, tau, x, N, j):

    total_paths = np.empty(N+1)

    for i in range(j):

        time, path = randompathgen(x_0, tau, x, N).T

        total_paths = np.vstack((path, total_paths))

    return time, total_paths[:-1]



#combines randompathgen and array of several paths into one function

def all_paths(x_0, tau, x, N, k, rng=None, bound = 2):
    if rng is None:
        rng = np.random.default_rng()

    time = np.linspace(0, tau, N + 1)

    paths = np.empty((k, N + 1))
    paths[:, 0] = x_0
    paths[:, -1] = x
    paths[:, 1:-1] = rng.uniform(-bound, bound, (k, N - 1))

    return time, paths



time, paths = all_paths(x_0, tau, x, N, k, bound = 2)
#print(time)
#print(paths)

#path_energies = summed_energy(m, paths, tau, N, V)

#weighted array of paths:

def weights(totalpaths, summedenergy):

    path_energy = summed_energy(m, totalpaths[])

    return 

'''

#NO LONGER NEED RANDOM PATH GEN BEFORE INTEGRATION, AS VEGAS SORTS THAT DURING MONTE CARLO INTEGRATION IN PROPAGATOR.



def propagator_integrand(m, x_ini, x_fin, tau, N, V):


    @vegas.batchintegrand   #allows vegas to pass many points at once
    def path_weights(y):

        n = y.shape[0]  #selects number of points vegas has chosen to evaluate per path
        paths = np.empty((n, N + 1))
        paths[:, 0] = x_ini
        paths[:, -1] = x_fin
        paths[:, 1:-1] = y   #the N-1 interior points that vegas uses

        v_avg = (paths[:, 1:] - paths[:, :-1])/(tau/N)
        midpoint = 0.5 * (paths[:, 1:] + paths[:, :-1])
        summed_energy = np.sum((0.5*m*(v_avg**2))+ V(midpoint), axis=1)
        return np.exp(-(tau/N)*summed_energy)

        '''
        weights = np.empty(k)

        for i in range(k):

            weights[i] = np.exp(-(tau/N)*summed_energy(m, paths[i][:], tau, N, V))

        return weights
        '''
    return path_weights

#weights = path_weights(m, paths, tau, N, V, k)

#print(weights)


#Propagator function, G:


def propagator(N, m, x_ini, x_fin, tau, V, bound):     #technically G/A not G here


    integ = vegas.Integrator([[-bound,bound]]*(N-1))  #vegas does multidimensional monte carlo integration

    path_weights = propagator_integrand(m, x_ini, x_fin, tau, N, V)

    integ(path_weights, nitn=10, neval=20000)   #warm-up run

    result = integ(path_weights, nitn=10, neval=20000)

    return result

result = propagator(N, m, x_0, x, tau, V, bound)


#print(result.summary())
#print(result.mean, result.sdev)


print(result)
