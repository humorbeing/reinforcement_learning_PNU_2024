
file_path = 'logs/f8.log'
save_name = 'f8'



with open(file_path, 'r') as f:
    _log = f.readlines()


# log_ = _log[1:]

def get_n(s):
    _, _round, _, _episode, _, _score, _, _epsilon, _ = s.split('|')
    round_ = int(_round)
    episode = int(_episode)
    score = float(_score)
    epsilon = float(_epsilon)
    return round_, episode, score, epsilon

return_matrix = []
epsilon_matrix = []

for r in range(999):
    retrun_list = []
    epsilon_list = []
    for episode in range(1000):
        index = r * 1000 + episode
        ep_log = _log[index]
        
        round_, episode_, score, epsilon = get_n(ep_log)
        assert round_ == r
        assert episode_ == episode
        

        retrun_list.append(score)   
        epsilon_list.append(epsilon)     
        print(f'{r}-{episode}')
    
    return_matrix.append(retrun_list)
    epsilon_matrix.append(epsilon_list)


import numpy as np

return_np = np.array(return_matrix)
epsilon_np = np.array(epsilon_matrix)


np.save(f'logs/{save_name}_return.npy', return_np)
np.save(f'logs/{save_name}_epsilon.npy', epsilon_np)


print("end")