import numpy as np
from flower_recovery_gaussian.rss_utils import add_pairwise_noise

'''
check_noise.py is to check if the add_pairwise_noise function works correctly; a.k.a, 
server aggregation = 0.0

File separate from flower framework
'''

def fedavg(params_list, n_list):
    N = sum(n_list)
    out = [np.zeros_like(w) for w in params_list[0]]
    for ws, n in zip(params_list, n_list):
        for k in range(len(out)):
            out[k] += (n / N) * ws[k]
    return out

# fake “weights”
w1 = np.ones((2,2), dtype=np.float32)
w2 = np.ones((3,), dtype=np.float32)
true = [w1, w2]

client_ids = [0,1]
n_i = [100, 50]
participants = [0,1]
round_no = 1
noise_std = 0.5

masked = []
for cid, n in zip(client_ids, n_i):
    masked.append(add_pairwise_noise(true, cid, participants, round_no, noise_std, 1.0/n))

avg_true   = fedavg([true,true], n_i)
avg_masked = fedavg(masked, n_i)

print("Max diff:", max(np.max(np.abs(a-b)) for a,b in zip(avg_true, avg_masked)))
