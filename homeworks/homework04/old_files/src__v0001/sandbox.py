td_path = '/home/ray/workspace/codes/playground/reinforcement_learning_PNU_2024/homeworks/homework04/results/logs/td.npy'
mc_every_path = '/home/ray/workspace/codes/playground/reinforcement_learning_PNU_2024/homeworks/homework04/results/logs/every.npy'
mc_first_path = '/home/ray/workspace/codes/playground/reinforcement_learning_PNU_2024/homeworks/homework04/results/logs/first.npy'

import numpy as np
import scipy

tt = np.load(td_path)
ee = np.load(mc_every_path)
ff = np.load(mc_first_path)


t_mean = np.mean(tt, 0)
e_mean = np.mean(ee, 0)
f_mean = np.mean(ff, 0)

x = np.arange(1, 1000+1)
import matplotlib.pyplot as plt
plt.plot(x, e_mean, label='MC every visit')
plt.plot(x, f_mean, label='MC first visit')
plt.plot(x, t_mean, label='TD')

plt.legend()
plt.show()



td_path = '/home/ray/workspace/codes/playground/reinforcement_learning_PNU_2024/homeworks/homework04/results/logs/td.log'
mc_every_path = '/home/ray/workspace/codes/playground/reinforcement_learning_PNU_2024/homeworks/homework04/results/logs/everyvisit.log'
mc_first_path = '/home/ray/workspace/codes/playground/reinforcement_learning_PNU_2024/homeworks/homework04/results/logs/firstvisit.log'


with open(td_path, 'r') as f:
    td_log = f.readlines()

with open(mc_every_path, 'r') as f:
    every_log = f.readlines()

with open(mc_first_path, 'r') as f:
    first_log = f.readlines()


tl = td_log[1:]
el = every_log[1:]
fl = first_log[1:]

def get_n(s):
    _, r, _, e, _, score, _ = s.split('|')
    r = int(r)
    e = int(e)
    score = float(score)
    score = round(score, 5)
    return e, score

tmatrix = []
ematrix = []
fmatrix = []
# for r in range(1000):
for r in range(999):
    t_list = []
    e_list = []
    f_list = []
    for episode in range(1000):
        index = r * 1000 + episode
        tt = tl[index]
        ee = el[index]
        ff = fl[index]
        tepisode, tscore = get_n(tt)
        assert tepisode == episode
        eepisode, escore = get_n(ee)
        assert eepisode == episode
        fepisode, fscore = get_n(ff)
        assert fepisode == episode

        t_list.append(tscore)
        e_list.append(escore)
        f_list.append(fscore)
        print(f'{r}-{episode}')
    
    tmatrix.append(t_list)
    ematrix.append(e_list)
    fmatrix.append(f_list)


import numpy as np

ttt = np.array(tmatrix)
eee = np.array(ematrix)
fff = np.array(fmatrix)

np.save('td.npy', ttt)
np.save('every.npy', eee)
np.save('first.npy', fff)
print("end")