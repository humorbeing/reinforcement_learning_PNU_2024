import h5py
h5_path = '/home/ray/workspace/codes/playground/reinforcement_learning_PNU_2024/homeworks/project01/dataset/small/CMU_001_01.hdf5'
dset = h5py.File(h5_path, "r")
print("Expert actions from first rollout episode:")
# print(dset["CMU_002_01-0-92/0/actions"][...])
print('e')