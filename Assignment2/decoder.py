import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--value-policy",type=str,default="/")
parser.add_argument("--states",type=str,default="/")

args = parser.parse_args()

state_file = args.states
value_file = args.value_policy

act_list = [0, 1, 2, 4, 6]

with open(state_file, 'r') as f:
    state_list = f.readlines()
    
with open(value_file, 'r') as f:
    value_list = f.readlines()
    
for i in range(2, len(value_list)):
    state_value = float(value_list[i].split()[0])
    state_action = act_list[int(value_list[i].split()[1])]
    state_name = state_list[i-2].strip()
    
    value = str("{:.6f}".format(state_value))
    print(str(state_name), str(state_action), str(value))
    
