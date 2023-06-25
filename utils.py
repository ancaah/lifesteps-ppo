import numpy as np

# new iterative functions

def calc_advantages(T, adv, rewards, values, next_val, gamma, lmbda, terminated, truncated):
    # T should have been passed as T-1 because of array indexing
    # Note: terminated and truncated are not temporal synchronized with other vectors
    # terminated[t] is true if at time t, with obs[t+1], the episode is terminated
    # values[t] is referred to obs[t], while next_val is referred to obs T+1
    done = 1- np.maximum(terminated, truncated)

    # first "iteration", to fill the whole advantage vector with significant values
    # otherwise we should put 0 at the last column
    delta = rewards[T] + gamma*next_val*done[T] - values[T]
    adv[T] = last_v = delta

    for t in reversed(range(T-1)):
        delta = rewards[t] + gamma*values[t+1]*done[t] - values[t]
        adv[t] = last_v = gamma*lmbda*done[t]*last_v + delta

    return adv


def calc_returns(T, ret, rewards, next_val, gamma, terminated, truncated):
    done = 1- np.maximum(terminated, truncated)

    ret[T] = last_r = rewards[T] + gamma*next_val*done[T]

    for t in reversed(range(T-1)):
        ret[t] = last_r = rewards[t] + gamma*last_r*done[t]

    return ret


##############################################################
#   Recursive and
#   Deprecated methods
#
#
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

def calc_returns_list(T, t, rewards, gamma, terminated, truncated):
    r = []

    # calculating the t(th) factor
    #done = terminated[T+1] + truncated[T+1]
    # should have nextval...
    
    p = rewards[T]

    calc_returns_r(T-1, t, rewards, gamma, terminated, truncated, p, r)
    r.append(p)
    return r

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
    
def incremental_mean(mean, val, n):
    return (mean*n + val)/(n+1)