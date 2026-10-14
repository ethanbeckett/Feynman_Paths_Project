from random import randint
import numpy as np
import vegas

#INITIAL VALUES
x_0 = 0
tau = 4
x = 10
N = 7
m = 1
k = 10 #number of paths generated
bound = 10

def V(x):
    return 0



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

'''

#Summed Energy Along Path:


def summed_energy(m, paths, tau, N, V):     #V is potential and is a function of position/the midpoint between x_j and x_(j+1)

    v_avg = (paths[1:] - paths[:1])/(tau/N)
    midpoint = 0.5 * (paths[1:] + paths[:1])

    return np.sum((0.5*m*(v_avg**2))+ V(midpoint))

'''

#array of several paths

def all_paths(x_0, tau, x, N, j):

    total_paths = np.empty(N+1)

    for i in range(j):

        time, path = randompathgen(x_0, tau, x, N).T

        total_paths = np.vstack((path, total_paths))

    return time, total_paths[:-1]

'''

#combines randompathgen and array of several paths into one function

def all_paths(x_0, tau, x, N, k, rng=None, bound = 10000):
    if rng is None:
        rng = np.random.default_rng()

    time = np.linspace(0, tau, N + 1)

    paths = np.empty((k, N + 1))
    paths[:, 0] = x_0
    paths[:, -1] = x
    paths[:, 1:-1] = rng.uniform(-bound, bound, (k, N - 1))

    return time, paths



time, paths = all_paths(x_0, tau, x, N, k, bound=20)
#print(time)
#print(paths)

#path_energies = summed_energy(m, paths, tau, N, V)

#weighted array of paths:
'''
def weights(totalpaths, summedenergy):

    path_energy = summed_energy(m, totalpaths[])

    return 
'''

def path_weights(m, paths, tau, N, V, k):

    weights = np.empty(k)

    for i in range(k):

        weights[i-1] = np.exp(-(tau/N)*summed_energy(m, paths[i-1][:], tau, N, V))

    return weights

weights = path_weights(m, paths, tau, N, V, k)

print(weights)


#Propagator function, G:

'''
def propagator(m, x_0, tau, N, x, V, L):      #technically G/A not G here

    epsilon = tau/N

    time, path = randompathgen(x_0, tau, x, N).T   #transposes so can unpack rows successfully

    path_energy = summed_energy(m, path, tau, N, V)

    weight = np.exp(-(epsilon * path_energy))

    integ = vegas.Integrator([-L,L]*(N-1))

    result = integ(weight, nitn=10, neval=1000)

    return result

'''