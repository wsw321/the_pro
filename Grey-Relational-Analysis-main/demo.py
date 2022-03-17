import numpy as np

a = np.array([[1, 1, 1],
              [2, 2, 2],
              [3, 3, 3],
              [4, 9, 4]])
failed_dict = {'camera': set(), 'radar': set()}
failed_dict['camera'].add((80,8))
failed_dict['camera'].add((9,45))
failed_dict['radar'].add(())
print(failed_dict)
failed_camera_set = {i for i in range(a.shape[0])}
print(failed_camera_set)

a = {i[0] for i in failed_dict['camera']}
print(a)