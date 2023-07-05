import numpy as np

# new iterative functions

# calculates advantages including the last term, using the first state-value outside of
# the scope of the batch. This method was inspired by the CleanRL implementation of PPO,
# after an initial recursive design.

def calc_advantages(T, adv, rewards, values, next_val, gamma, lmbda, terminated, truncated):
    # support array used to remove from the sum of a delta the sum of the successive delta
    done = 1- np.maximum(terminated, truncated)

    # first "iteration", to fill the whole advantage vector with significant values
    # otherwise we should put 0 at the last column
    last_v = rewards[T-1] + gamma*next_val*done[T-1] - values[T-1]
    adv[T-1] = last_v

    for t in reversed(range(T-1)):
        delta = rewards[t] + gamma*values[t+1]*done[t] - values[t]
        adv[t] = last_v = delta + gamma*lmbda*done[t]*last_v

    return adv

# calculates returns including the last term, using the first state-value outside of
# the scope of the batch

def calc_returns(T, ret, rewards, next_val, gamma, terminated, truncated):
    # support array used to remove from the sum of a return the sum of next discounted rewards
    done = 1- np.maximum(terminated, truncated)

    for t in reversed(range(T)):
        ret[t] = last_r = rewards[t] + gamma*next_val*done[t]
        next_val = last_r

    return ret


# performs an incremental mean update

def incremental_mean(mean, val, n):
    return (mean*n + val)/(n+1)

##############################################################
#   First (and recursive) set of methods for calculation of
#   advantages and returns. Not used in the final code
#   but still working
#

# advantages calculation: variant which uses the first state-value outside of the scope of the batch
#                           for the last advantage calculation

def calc_adv_list_wlast(T, t, rewards, values, gamma, lmbda, terminated, truncated, next_val):
    r = []

    # calculating the t(th) factor
    #done = terminated[T+1] + truncated[T+1]
    # should have nextval...
    done = 1- max(terminated[T], truncated[T])
    p = rewards[T] + gamma*next_val*done - values[T]

    calc_delta_list_r(T-1, t, rewards, values, gamma, lmbda, terminated, truncated, p, r)
    r.append(p)
    return r

# advantages calculation: variant which puts the last term to 0

def calc_adv_list(T, t, rewards, values, gamma, lmbda, terminated, truncated):
    r = []

    # calculating the t(th) factor
    #done = terminated[T+1] + truncated[T+1]
    # should have nextval...
    
    # here the first calculated advantage, which is on time T, is set to 0 instead of approximating it
    p = 0

    calc_delta_list_r(T-1, t, rewards, values, gamma, lmbda, terminated, truncated, p, r)
    r.append(p)
    return r
    

# recursive support function for calc_adv_list and calc_adv_list_wlast

def calc_delta_list_r(T, t, rewards, values, gamma, lmbda, terminated, truncated, p, r):
    # calculating the t(th) factor
    done = 1- max(terminated[T], truncated[T])
    p = p*gamma*lmbda*done + rewards[T] + gamma*values[T + 1]*done - values[T]

    if T == t:
        r.append(p)
        return p
    else:
        calc_delta_list_r(T-1, t, rewards, values, gamma, lmbda, terminated, truncated, p, r)
        r.append(p)
        return p


# calculates returns just considering the actual rewards

def calc_returns_list(T, t, rewards, gamma, terminated, truncated):
    r = []

    # calculating the t(th) factor
    #done = terminated[T+1] + truncated[T+1]
    # should have nextval...
    
    p = rewards[T]

    calc_returns_r(T-1, t, rewards, gamma, terminated, truncated, p, r)
    r.append(p)
    return r


# support recursive function for calc_returns_list

def calc_returns_r(T, t, rewards, gamma, terminated, truncated, p, r):
    # calculating the t(th) factor
    done = 1 - max(terminated[T], truncated[T])
    p = p*gamma*done + rewards[T]

    if T == t:
        r.append(p)
        return p
    else:
        calc_returns_r(T-1, t, rewards, gamma, terminated, truncated, p, r)
        r.append(p)
        return p
