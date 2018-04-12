import multiprocessing
from random import randint

EXPERIMENT_SIZE = 100000
NUM_COINS = 1000
TOSSES_PER_COIN = 10


v_1_all = multiprocessing.Array('f', [ 0 for i in range(EXPERIMENT_SIZE) ], lock=False)
v_min_all = multiprocessing.Array('f', [ 0 for i in range(EXPERIMENT_SIZE) ], lock=False)
v_rand_all = multiprocessing.Array('f', [ 0 for i in range(EXPERIMENT_SIZE) ], lock=False)

def experiment(ex):

	global v_1_all, v_min_all, v_rand_all
	
	# 1 - heads
	# 0 - tails
	coins = [ 0 for i in range(NUM_COINS) ]

	c_min = TOSSES_PER_COIN

	for c in range(len(coins)):

		for i in range(TOSSES_PER_COIN):

			if randint(0, 1) == 1:
				coins[c] += 1


		if coins[c] < c_min:
			c_min = coins[c]

	v_1_all[ex] = coins[0] / float(TOSSES_PER_COIN)
	v_min_all[ex] = c_min / float(TOSSES_PER_COIN)
	v_rand_all[ex] = coins[ randint(0, NUM_COINS-1) ] / float(TOSSES_PER_COIN)


pool = multiprocessing.Pool(processes=8, initargs=(v_1_all, v_min_all, v_rand_all))
pool.map(experiment, [ i for i in range(EXPERIMENT_SIZE) ])


print("v_1 average: ", sum(v_1_all) / float(EXPERIMENT_SIZE))
print("v_min average: ", sum(v_min_all) / float(EXPERIMENT_SIZE))
print("v_rand average: ", sum(v_rand_all) / float(EXPERIMENT_SIZE))