e10_path = 'logs/e10_return.npy'
f1_path = 'logs/f7_return.npy'
f2_path = 'logs/f8_return.npy'
f3_path = 'logs/f10_return.npy'
f4_path = 'logs/f11_return.npy'
f5_path = 'logs/f6_return.npy'
path_list = [
    ['logs/e10_epsilon.npy', 'epsilon 10%'],
    # ['logs/f1_epsilon.npy', 'line 0.9-0.05'],
    # ['logs/f2_epsilon.npy', 'J-curve 0.99-0.05'],
    # ['logs/f3_epsilon.npy', 'f3'],
    # ['logs/f4_epsilon.npy', 's-curve 0.6-0.05'],
    # ['logs/f5_epsilon.npy', 'f5'],
    ['logs/f6_epsilon.npy', 'f6'],
    ['logs/f7_epsilon.npy', 'f7'],
    ['logs/f8_epsilon.npy', 'f8'],    
    # ['logs/f10_epsilon.npy', 'f10'],
    ['logs/f11_epsilon.npy', 'f11'],
]


import numpy as np
import scipy

np_list = []

for i in path_list:
    np_list.append(np.load(i[0]))


cut = 985

# t_mode = scipy.stats.mode(tt, 0)[0][0]
# e_mode = scipy.stats.mode(ee, 0)[0][0]
# f_mode = scipy.stats.mode(ff, 0)[0][0]


# t_skew = scipy.stats.skew(tt, 0)
# e_skew = scipy.stats.skew(ee, 0)
# f_skew = scipy.stats.skew(ff, 0)
mean_list = []
for i in np_list:
    mean_list.append(np.mean(i, 0))


# t_mean_cut = np.mean(tt, 0)[:cut]
# e_mean_cut = np.mean(ee, 0)[:cut]
# f_mean_cut = np.mean(ff, 0)[:cut]

# t_median = np.median(tt, 0)
# e_median = np.median(ee, 0)
# f_median = np.median(ff, 0)

# t_25 = np.quantile(tt, 0.35, 0)
# e_25 = np.quantile(ee, 0.35, 0)
# f_25 = np.quantile(ff, 0.35, 0)

# t_75 = np.quantile(tt, 0.65, 0)
# e_75 = np.quantile(ee, 0.65, 0)
# f_75 = np.quantile(ff, 0.65, 0)

# t_05 = np.quantile(tt, 0.15, 0)
# e_05 = np.quantile(ee, 0.15, 0)
# f_05 = np.quantile(ff, 0.15, 0)

# t_95 = np.quantile(tt, 0.85, 0)
# e_95 = np.quantile(ee, 0.85, 0)
# f_95 = np.quantile(ff, 0.85, 0)

# t_25 = np.percentile(tt, 25, 0)
# e_25 = np.percentile(ee, 0.35, 0)
# f_25 = np.percentile(ff, 0.35, 0)

# t_75 = np.percentile(tt, 75, 0)
# e_75 = np.percentile(ee, 0.65, 0)
# f_75 = np.percentile(ff, 0.65, 0)

# t_05 = np.percentile(tt, 5, 0)
# e_05 = np.percentile(ee, 0.15, 0)
# f_05 = np.percentile(ff, 0.15, 0)

# t_95 = np.percentile(tt, 95, 0)
# e_95 = np.percentile(ee, 0.85, 0)
# f_95 = np.percentile(ff, 0.85, 0)

# t_std = np.std(tt, 0)
# e_std = np.std(ee, 0)
# f_std = np.std(ff, 0)

x_cut = np.arange(1, cut+1)
x = np.arange(1, 1000+1)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# t_1 = smooth(t_mean,3)[:cut]
# t_2 = smooth(t_mean,15)[:cut]
# t_3 = smooth(t_mean,30)[:cut]

smooth_list = []
for i in mean_list:
    smooth_list.append(smooth(i,15)[:cut])


# e_1 = smooth(e_mean,3)[:cut]
# e_2 = smooth(e_mean,15)[:cut]
# e_3 = smooth(e_mean,30)[:cut]

# f_1 = smooth(f_mean,3)[:cut]
# f_2 = smooth(f_mean,15)[:cut]
# f_3 = smooth(f_mean,30)[:cut]

import matplotlib.pyplot as plt

# plt.plot(x, t_mean)
# plt.plot(x_cut, t_1)
# plt.plot(x_cut, t_2)
# plt.plot(x_cut, t_3)

# plt.plot(x_cut, e_mean_cut, label='MC every visit')
# plt.plot(x_cut, f_mean_cut, label='MC first visit')
# plt.plot(x_cut, t_mean_cut, label='TD')

# for i, s in enumerate(smooth_list):
#     plt.plot(x_cut, s, label=path_list[i][1])

for i, s in enumerate(mean_list):
    plt.plot(x, s, label=path_list[i][1])
# plt.plot(x, e_mean, label='MC every visit')
# plt.plot(x, f_mean, label='MC first visit')
# plt.plot(x, t1_mean, label='e10')
# plt.plot(x, t3_mean, label='f1')
# plt.plot(x, t5_mean, label='f2')
# plt.plot(x, t7_mean, label='f3')
# plt.plot(x, f4_mean, label='f4')
# plt.plot(x, f5_mean, label='f5')


# plt.plot(x, e_std, label='MC every visit')
# plt.plot(x, f_std, label='MC first visit')
# plt.plot(x, t_std, label='TD')


# plt.plot(x, e_skew, label='MC every visit')
# plt.plot(x, f_skew, label='MC first visit')
# plt.plot(x, t_skew, label='TD')



# plt.plot(x, e_mode, label='MC every visit')
# plt.plot(x, f_mode, label='MC first visit')
# plt.plot(x, t_mode, label='TD')
plt.legend()
plt.ylim((0,1))
plt.title('Epsilon Change Curve')
# plt.plot(x, t_median)
# plt.plot(x, e_median)
# plt.plot(x, f_median)

# plt.plot(x, t_median, color='#CC4F2B')
# plt.fill_between(x, t_25, t_75,
#                  alpha=0.3, edgecolor='#CC4F1B', facecolor='#FF9848')
# plt.fill_between(x, t_05, t_95,
#                  alpha=0.1, edgecolor='#CC4F1B', facecolor='#FF9848')


# plt.plot(x, e_median, color='#1B2ACC')
# plt.fill_between(x, e_25, e_75,
#                  alpha=0.3, edgecolor='#1B2ACC', facecolor='#089FFF')
# plt.fill_between(x, e_05, e_95,
#                  alpha=0.1, edgecolor='#1B2ACC', facecolor='#089FFF')


# plt.plot(x, f_median, color='#3F7F4C')
# plt.fill_between(x, f_25, f_75,
#                  alpha=0.3, edgecolor='#3F7F4C', facecolor='#7EFF99')
# plt.fill_between(x, f_05, f_95,
#                  alpha=0.1, edgecolor='#3F7F4C', facecolor='#7EFF99')
plt.show()

print('end')