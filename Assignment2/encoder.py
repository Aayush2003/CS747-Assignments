import argparse
parser = argparse.ArgumentParser()

def find(l, element):
    if(element not in l):
        return -2  # When added with 2, returns state 0 (losing end state)
    else:
        return l.index(element)
    
def store(trans, s1, a, s2, r, p):
    if((s1,a,s2) not in trans.keys()):
        trans[(s1,a,s2)] = [r, p]
    else:
        # Different outcome for same action, still takes you to same state
        trans[(s1,a,s2)][1] += p 
        
def PlayerB(statelist, curr_state, next_state, q, i, a, prob, trans):
    # All sequences played out by B
    bb = int(next_state[0:2])
    rr = int(next_state[2::])
    pout = q
    p0 = (1-q)/2
    p1 = (1-q)/2
    newprob = 0
    if(bb == 0):   # Player A has lost game
        store(trans, i+2, a, 0, 0, prob)
        return
    while bb > 0:
        # Out
        store(trans, i+2, a, 0, 0, prob*pout)
        # 0 runs
        if (bb%6 == 1):
            if(bb == 1): # Last ball of game, then lose
                store(trans, i+2, a, 0, 0, prob*p0)
            else: # A back on strike, create transition
                next_state = str(bb - 1).zfill(2) + str(rr).zfill(2)
                store(trans, i+2, a, find(statelist, next_state)+2, 0, prob*p0)
        else: # B is still on strike, recursive call
            rr -= 0
            newprob = prob*p0
        # 1 run
        if (rr <= 1): # Win
            store(trans, i+2, a, 1, 1, prob*p1)
            if(bb%6 == 1): # B can no longer stall, exit sequence
                break
        else:
            if (bb%6 == 1):
                if(bb == 1): # Last ball of game, then lose
                    store(trans, i+2, a, 0, 0, prob*p1)
                else: # B is still on strike, recursive call
                    rr -= 1
                    newprob = prob*p1
            else: # A back on strike, create transition
                next_state = str(bb - 1).zfill(2) + str(rr-1).zfill(2)
                store(trans, i+2, a, find(statelist, next_state)+2, 0, prob*p1)
        prob = newprob
        bb -= 1
        
    return
   

parser.add_argument("--states",type=str,default="/")
parser.add_argument("--parameters",type=str,default="/")
parser.add_argument("--q",type=float,default=0.25)

args = parser.parse_args()

state_file = args.states
param_file = args.parameters
q = args.q

with open(state_file, 'r') as f:
    statelist = f.readlines()
    
for i in range(len(statelist)):
    statelist[i] = statelist[i].strip()
    
p1_param = dict()
    
with open(param_file, 'r') as f:
    lines = f.readlines()
    for i in range(1, 6):
        linelist = lines[i].split()
        p1_param[i-1] = [float(linelist[1]), float(linelist[2]), float(linelist[3]), float(linelist[4]), float(linelist[5]), float(linelist[6]), float(linelist[7])]
    
print("numStates", len(statelist) + 2)
print("numActions", 5)
print("end", 0, 1)   # 0 is losing end state, 1 is winning end state

trans = dict()

for i in range(len(statelist)):
    curr_state = statelist[i]
    bb = int(curr_state[0:2])
    rr = int(curr_state[2::])
    
    for j in range(5): # Over all actions
        # Out
        store(trans, i+2, j, 0, 0, p1_param[j][0])
        # 0 runs
        next_state = str(bb - 1).zfill(2) + str(rr).zfill(2)
        if(bb%6 == 1): # B on strike
            PlayerB(statelist, curr_state, next_state, q, i, j, p1_param[j][1], trans)
        else:
            store(trans, i+2, j, find(statelist, next_state)+2, 0, p1_param[j][1])
        # 1 run
        next_state = str(bb - 1).zfill(2) + str(rr-1).zfill(2)
        if(rr <= 1): # Win
            store(trans, i+2, j, 1, 1, p1_param[j][2])
        else:
            if(bb%6 == 1): # A retains strike
                store(trans, i+2, j, find(statelist, next_state)+2, 0, p1_param[j][2])
            else:
                PlayerB(statelist, curr_state, next_state, q, i, j, p1_param[j][2], trans)
        # 2 runs
        next_state = str(bb - 1).zfill(2) + str(rr-2).zfill(2)
        if(rr <= 2): # Win
            store(trans, i+2, j, 1, 1, p1_param[j][3])
        else:
            if(bb%6 == 1): # B on strike
                PlayerB(statelist, curr_state, next_state, q, i, j, p1_param[j][3], trans)
            else:
                store(trans, i+2, j, find(statelist, next_state)+2, 0, p1_param[j][3])
        # 3 runs
        next_state = str(bb - 1).zfill(2) + str(rr-3).zfill(2)
        if(rr <= 3): # Win
            store(trans, i+2, j, 1, 1, p1_param[j][4])
        else:
            if(bb%6 == 1): # A retains strike
                store(trans, i+2, j, find(statelist, next_state)+2, 0, p1_param[j][4])
            else:
                PlayerB(statelist, curr_state, next_state, q, i, j, p1_param[j][4], trans)
        # 4 runs
        next_state = str(bb - 1).zfill(2) + str(rr-4).zfill(2)
        if(rr <= 4): # Win
            store(trans, i+2, j, 1, 1, p1_param[j][5])
        else:
            if(bb%6 == 1): # B on strike
                PlayerB(statelist, curr_state, next_state, q, i, j, p1_param[j][5], trans)
            else:
                store(trans, i+2, j, find(statelist, next_state)+2, 0, p1_param[j][5])
        # 6 runs
        next_state = str(bb - 1).zfill(2) + str(rr-6).zfill(2)
        if(rr <= 6): # Win
            store(trans, i+2, j, 1, 1, p1_param[j][6])
        else:
            if(bb%6 == 1): # B on strike
                PlayerB(statelist, curr_state, next_state, q, i, j, p1_param[j][6], trans)
            else:
                store(trans, i+2, j, find(statelist, next_state)+2, 0, p1_param[j][6])
                

for i in trans.keys():
    if(trans[i][1] != 0):
        print("transition", i[0], i[1], i[2], trans[i][0], trans[i][1])

print("mdptype", "episodic")
print("discount", 1)