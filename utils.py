def calc_adv_list_wlast(T, t, rewards, values, gamma, lmbda, terminated, truncated, next_val):
    r = []

    # calculating the t(th) factor
    #done = terminated[T+1] + truncated[T+1]
    # should have nextval...
    
    p = rewards[T] + gamma*next_val - values[T]

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
    done = 1- (terminated[T+1] + truncated[T+1])
    p = p*gamma*lmbda + rewards[T] + gamma*values[T + 1]*done - values[T]

    if T == t:
        r.append(p)
        return p
    else:
        calc_delta_list_r(T-1, t, rewards, values, gamma, lmbda, terminated, truncated, p, r)
        r.append(p)
        return p

def calc_returns(T, t, rewards, gamma, terminated, truncated):
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
    done = 1- (terminated[T] + truncated[T])
    p = p*gamma*done + rewards[T]

    if T == t:
        r.append(p)
        return p
    else:
        calc_returns_r(T-1, t, rewards, gamma, terminated, truncated, p, r)
        r.append(p)
        return p