import numpy as np
import pulp
import argparse
parser = argparse.ArgumentParser()

def PolicyEval(n, a, terminal, transitions, g, pi):
    v = np.zeros(n)
    vnew = np.ones(n)
    possible_trans = transitions.keys()
    while np.abs(v - vnew).max() > 1e-12:
        v = np.copy(vnew)
        for i in range(n):
            if (i in terminal):
                vnew[i] = 0                    # Terminal State
                continue
            sumv = 0
            for j in range(n):
                if((i,int(pi[i]),j) in possible_trans):
                    # Transition (i,pi,j) has non-zero probability
                    r = transitions[(i,int(pi[i]),j)][0]
                    t = transitions[(i,int(pi[i]),j)][1]
                    sumv += t*(r + g*v[j])
            vnew[i] = sumv
    v = vnew
    return v

def VI(n, a, terminal, transitions, mdptype, g):
    v = np.zeros(n)
    pi = np.zeros(n)
    vnew = np.ones(n)
    possible_trans = transitions.keys()
    # Value Iteration
    while np.abs(v - vnew).max() > 1e-12:
        v = np.copy(vnew)
        for i in range(n):
            if (i in terminal):
                vnew[i] = 0                    # Terminal State
                continue
            maxsum = 0
            for j in range(a):
                newsum = 0
                for k in range(n):
                    if((i,j,k) in possible_trans):
                        # Transition (i,j,k) has non-zero probability
                        r = transitions[(i,j,k)][0]
                        t = transitions[(i,j,k)][1]
                        newsum += t*(r + g*v[k])
                if(newsum > maxsum):
                    # Action j maximizes value function
                    maxsum = newsum
                    pi[i] = j
            vnew[i] = maxsum

    v = vnew
    for i in range(n):
        value = str("{:.6f}".format(v[i]))
        action = str(int(pi[i]))
        print(value + "\t" + action + "\n", end = "")
        
        
def HPI(n, a, terminal, transitions, mdptype, g):
    v = np.zeros(n)
    pi = np.zeros(n)
    possible_trans = transitions.keys()
    isImprovable = True
    
    while isImprovable:
        v = PolicyEval(n, a, terminal, transitions, g, pi)
        isImprovable = False
        
        for i in range(n):
            if (i in terminal):
                continue
            improvable = list()
            for j in range(a):
                sumq = 0
                for k in range(n):
                    if((i,j,k) in possible_trans):
                        # Transition (i,j,k) has non-zero probability
                        r = transitions[(i,j,k)][0]
                        t = transitions[(i,j,k)][1]
                        sumq += t*(r + g*v[k])
                if(sumq-v[i]>1e-6):
                    improvable.append(j)
                    
            if(len(improvable) > 0):
                pi[i] = np.random.choice(improvable)
                isImprovable = True
    
    for i in range(n):
        value = str("{:.6f}".format(v[i]))
        action = str(int(pi[i]))
        print(value + "\t" + action + "\n", end = "")  
        
        
        
def LP(n, a, terminal, transitions, mdptype, g):
    possible_trans = transitions.keys()
    v = pulp.LpVariable.dicts("Value Function", np.arange(0,n), None, None)
    prob = pulp.LpProblem("LinearProgram", pulp.LpMaximize)
    objective = 0
    for i in range(n):
        objective += -v[i]
        if(i in terminal):
            prob += v[i] >= 0
            continue
        for j in range(a):
            rhs = 0
            lhs = v[i]
            for k in range(n):
                if((i,j,k) in possible_trans):
                    # Transition (i,j,k) has non-zero probability
                    r = transitions[(i,j,k)][0]
                    t = transitions[(i,j,k)][1]
                    rhs += t*r
                    lhs += -t*g*v[k]
            # Add constraint
            prob += lhs >= rhs
    # Add Objective
    prob += objective
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    for i in range(n):
        value = str("{:.6f}".format(pulp.value(v[i])))
        # Finding action which maximizes Bellman Equation for current state 
        if(i not in terminal):
            maxsum = 0
            for j in range(a):
                newsum = 0
                for k in range(n):
                    if((i,j,k) in possible_trans):
                        # Transition (i,j,k) has non-zero probability
                        r = transitions[(i,j,k)][0]
                        t = transitions[(i,j,k)][1]
                        newsum += t*(r + g*pulp.value(v[k]))
                if(newsum > maxsum):
                    # Action j maximizes value function
                    maxsum = newsum
                    action = str(j)
        else:
            action = '0'
        print(value + "\t" + action + "\n", end = "")  


parser.add_argument("--mdp",type=str,default="/")
parser.add_argument("--algorithm",type=str,default="vi")
parser.add_argument("--policy",type=str,default="")

args = parser.parse_args()

file_name = args.mdp
algo = args.algorithm
pol_file = args.policy

if(pol_file == ""):
    isPolicyEval = False
else:
    isPolicyEval = True
    
with open(file_name, 'r') as f:
    lines = f.readlines()
    numStates = int((lines[0].split())[1])
    numActions = int((lines[1].split())[1])
    endState = []
    for i in lines[2].split():
        if (i != "end" and i != "-1"):
            endState.append(int(i))
    transitions = dict()
    i = 3
    while lines[i][0:10] == "transition":
        trans_info = lines[i].split()
        trans_key = tuple((int(trans_info[1]), int(trans_info[2]), int(trans_info[3])))  # s1 a s2
        trans_val = tuple((float(trans_info[4]), float(trans_info[5])))                  # r p
        transitions[trans_key] = trans_val
        i += 1
    mdptype = lines[i].split()[1]
    gamma = float(lines[i+1].split()[1])

if(isPolicyEval):
    with open(pol_file, 'r') as f:
        policy = f.readlines()
    
    v = PolicyEval(numStates, numActions, endState, transitions, gamma, policy)
    for i in range(numStates):
        value = str("{:.6f}".format(v[i]))
        action = str(int(policy[i]))
        print(value + "\t" + action + "\n", end = "")
else:
    if(algo == "vi"):
        VI(numStates, numActions, endState, transitions, mdptype, gamma)
    elif(algo == "hpi"):   
        HPI(numStates, numActions, endState, transitions, mdptype, gamma)
    elif(algo == "lp"):
        LP(numStates, numActions, endState, transitions, mdptype, gamma)

