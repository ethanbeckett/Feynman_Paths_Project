'''

1D PATHS

RANDOM PATH GENERATOR:

Input:
    {x_0 - initial position
    t_0 - initial imaginary time (=0)
    x - final position
    tau - final time
    N  - time points, indexed by j}

epsilon = tau/N   -- time point separation

tau_j = j*epsilon


at each time point path goes in different randomised direction i.e the integral for G - only in 1D for this project so far
must start at x_0 and end at x in j steps

just use random number generator?

generate array of random numbers, last entry MUST be x-x_(N-1)

Output:
    an array of x_j = the path


'''