from random import randint
import numpy as np
import vegas
import matplotlib.pyplot as plt
from numba import njit

#For Z can also treat as an N dimensional integral and use VEGAS treating dx as the Nth variable to integrate over. Will have to redefine propagator again.
#Trapezoid rule works well for Z if using already obtained propagator function as it is a 1D integral - so don't NEED VEGAS to do this.



#tunneling freezing, double well something phenomena
#maybe add whitening to paths generated from metropolis

#INITIAL VALUES
x_0 = 0
tau = 4
x = 0
N = 7
m = 1
k = 100 #number of paths generated
bound = 5
G_bound = 5
Z_bound = 5
l = 10  #number of points plotted 


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

        n = y.shape[0]  #selects number of paths vegas has chosen to evaluate per point
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

G_result = propagator(N, m, x_0, x, tau, V, bound) #outputs G value at the point x


#print(result.summary())
#print(result.mean, result.sdev)


#print(result)

'''

def Z(N, m, tau, V, G_bound, Z_bound): #G with x_ini = x_fin, integrated over x

    integ = vegas.Integrator([[-Z_bound,Z_bound]])  #vegas does multidimensional monte carlo integration, dim = 1 here

    @vegas.batchintegrand
    def G_values(y):

        n = y.shape[0]
        G_value = np.empty([n, 1])

        for i in range(n):

            G_value[i] = propagator(N, m, y[i, 0], y[i, 0], tau, V, G_bound).mean  #returns mean as a float = G/A

        return G_value
     
        

    integ(G_values, nitn=10, neval=20000)   #warm-up run

    Z_result = integ(G_values, nitn=10, neval=20000)

    

    return Z_result

Z_result = Z(N,m, tau, V, G_bound, Z_bound)   #maybe change Z_bound to equal G_bound?

print(Z_result.summary())

''' #takes far too long to compute using nested VEGAS integration

def Z(N, m, tau, V, G_bound, Z_bound): #G with x_ini = x_fin, integrated over x

    xs = np.linspace(-Z_bound, Z_bound, 35) #35 x values to integrate G(x,x) over
    G_values = np.empty_like(xs)
    G_errors = np.empty_like(xs)

    for i, x in enumerate(xs):
        G_values[i] = propagator(N, m, x, x, tau, V, G_bound).mean   #G/A
        G_errors[i] = propagator(N, m, x, x, tau, V, G_bound).sdev

    Z = np.trapezoid(G_values, xs)
    #Z_err = work out later

    return Z

Z_value = Z(N, m, tau, V, G_bound, Z_bound)

G_value = G_result.mean

#print(Z_value)

#print(G_value)



def phi_squared(G, Z): #can vary x using constants defined at top of code

    return G/Z

#print(phi_squared(G_value, Z_value))







def feynman_phi_squared_plot(l):  #l = number of points to evaluate

    xval = np.linspace(0, 2, l)
    Gs = np.empty(l)
    Z = Z_value

    for i, x in enumerate(xval):

        Gs[i] = propagator(N, m, x, x, tau, V, bound).mean

    yval = phi_squared(Gs, Z)

    return xval, yval


xval_feynman = feynman_phi_squared_plot(l)[0]
yval_feynman = feynman_phi_squared_plot(l)[1]


def SE_phi_squared(x):

    return ((np.exp(-(x**2)/2)/((np.pi)**(1/4)))**2)


SE_phi = SE_phi_squared(x)




def SE_phi_squared_plot(l):  #l = number of points to evaluate

    xval = np.linspace(0, 2, l)
    SE_phi_val = np.empty(l)

    for i, x in enumerate(xval):

        SE_phi_val[i] = SE_phi_squared(x)

    yval = SE_phi_val

    return xval, yval


xval_SE = SE_phi_squared_plot(l)[0]
yval_SE = SE_phi_squared_plot(l)[1]










#VEGAS scales poorly with N (time points/ integral dimension), whereas Metropolis algorithm performes better for higher N

#Use metropolis algorithm to optimise vegas algorithm by pre-tuning its adaptive grid layout with selected paths.


def randompathgen(x_0, tau, x, N):

    time = (tau/N)*np.arange(N+1)  #array for timepoints, includes t_0 = 0 and tau
    x_j = np.random.default_rng()  #random number generator for x_j values, faster than random module for large arrays - better statistical properties and no hidden global state
    path = np.empty(N+1)
    path[0] = x_0
    path[-1] = x
    path[1:-1] = x_j.uniform(-5,5,N-1)

    return np.column_stack((time, path))



def summed_energy(m, paths, tau, N, V):     #V is potential and is a function of position/the midpoint between x_j and x_(j+1)

    v_avg = (paths[1:] - paths[:-1])/(tau/N)
    midpoint = 0.5 * (paths[1:] + paths[:-1])

    return np.sum((0.5*m*(v_avg**2))+ V(midpoint))



def metropolis(x_0, tau, x, N, delta, m, V, n_sweeps, n_therm = 1000):    #n_therm thermalisation sweeps before starting to collect x values to train model. #delta*rng.uniform(-1,1) = small number perturbed by.
    
    path = randompathgen(x_0, tau, x, N)[:, 1]

    optimised_paths = np.empty((n_sweeps, N+1))

    for j in range(n_sweeps + n_therm):

        for i in range(1, N): 

            initial_energy = summed_energy(m, path, tau, N, V)
            perturbation = delta*(2*np.random.random() - 1)   # uniform in (-delta, delta) not just [0,1) which will bias results

            path[i] = path[i] + perturbation
            change_in_action = (tau/N)*(summed_energy(m, path, tau, N, V) - initial_energy)

            if change_in_action < 0 or np.random.random() < np.exp(-change_in_action):

                pass               
                
            else:

                path[i] = path[i] - perturbation

        if j >= n_therm:

            optimised_paths[j-n_therm] = path

    return optimised_paths #outputs every 50th path



def met_propagator(x_0, tau, x, N, delta, m, V, n_sweeps):

    paths = metropolis(x_0, tau, x, N, delta, m, V, n_sweeps)
    X = paths[::50, 1:N]   #every 50th path                       # shape (n_samples, N-1)

    path_weights = propagator_integrand(m, x_0, x, tau, N, V)
    fX = path_weights(X)

    # 3. Integration box per dimension: sample range plus a margin
    std = X.std(axis=0)
    lo = X.min(axis=0) - 2 * std
    hi = X.max(axis=0) + 2 * std
    amap = vegas.AdaptiveMap([[l, h] for l, h in zip(lo, hi)], ninc=50)

    # 4. Train the VEGAS map on the Metropolis samples
    amap.adapt_to_samples(X, fX, nitn=5)

    # 5. Integrate with fresh, independent VEGAS samples
    integ = vegas.Integrator(amap)
    integ(path_weights, nitn=5, neval=20000)      # short warm-up
    result = integ(path_weights, nitn=10, neval=20000)

    return result


def met_Z(N, m, tau, V, delta, n_sweeps, G_bound, Z_bound): #G with x_ini = x_fin, integrated over x

    xs = np.linspace(-Z_bound, Z_bound, 35) #35 x values to integrate G(x,x) over
    G_values = np.empty_like(xs)
    G_errors = np.empty_like(xs)

    for i, x in enumerate(xs):
        G_values[i] = met_propagator(x, tau, x, N, delta, m, V, n_sweeps).mean   #G/A
        G_errors[i] = met_propagator(x, tau, x, N, delta, m, V, n_sweeps).sdev

    met_Z = np.trapezoid(G_values, xs)
    #Z_err = work out later

    return met_Z

#Z_metropolis = met_Z(N, m, tau, V, 0.5, 40000, G_bound, Z_bound)


def phi_met_squared(G_metropolis, Z_metropolis):

    return G_metropolis/Z_metropolis


def met_feynman_phi_squared_plot(l):  #l = number of points to evaluate

    xval = np.linspace(0, 2, l)
    Gs = np.empty(l)
    Z = Z_metropolis

    for i, x in enumerate(xval):

        Gs[i] = met_propagator(x, tau, x, N, 0.5, m, V, 40000).mean

    yval = phi_met_squared(Gs, Z)

    return xval, yval


#xval_met = met_feynman_phi_squared_plot(l)[0]
#yval_met = met_feynman_phi_squared_plot(l)[1]


plt.errorbar(xval_feynman, yval_feynman, marker = '.', linestyle = 'none', color = 'b', label = 'Feynman Vegas')
plt.errorbar(xval_SE, yval_SE, marker = '.', linestyle = 'none', color = 'r', label = 'Schrodinger')
#plt.errorbar(xval_met, yval_met, marker = '.', linestyle = 'none', color = 'g', label = 'Feynman Metropolis')
plt.legend()
plt.show()


G_result_1 = propagator(N, m, 1, 1, tau, V, bound).mean
Feynman_1 = phi_squared(G_result_1, Z_value)
Schrodinger_1 = SE_phi_squared(1)
#G_met_1 = met_propagator(1, tau, 1, N, 0.5, m, V, 40000).mean
#Feynman_met_1 = phi_met_squared(G_met_1, Z_metropolis)


print("The value at x = 1 using the Feynman Path Integral Approach with VEGAS only is: ", Feynman_1)
#print("The value at x = 1 using the Feynman Path Integral Approach with Metropolis algorithm optimisation is:", Feynman_met_1)
print("The value at x = 1 using the Schrodinger analytical approach is: ", Schrodinger_1)






