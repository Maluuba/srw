
# e-greedy
ipython .\train.py -- -o bridge_len 10 -o explore_method egreedy -o target_eval True -o nb_episodes 2000000 -o saving_period 100000 -o printing_period 2000 -o writing_period 100 -o annealing False -o epsilon 0.1 -o gamma 0.99 -o alpha 0.1 -o folder_name egreedy_

# secure:
ipython .\train.py -- -o bridge_len 20 -o explore_method secure -o target_eval True -o nb_episodes 2000 -o saving_period 2000 -o printing_period 100 -o writing_period 10 -o gamma 1.0 -o epsilon 1.0 -o alpha 0.1 -o annealing_start_episode 1200 -o annealing True

# for L=25: annealing_start: 2600

# softmax:
ipython .\train.py -- -o bridge_len 14 -o explore_method softmax -o target_eval True -o nb_episodes 2000000 -o saving_period 20000 -o printing_period 2000 -o writing_period 100

# count-based
ipython .\train.py -- -o bridge_len 14 -o explore_method count -o target_eval True -o nb_episodes 2000000-o saving_period 50000 -o printing_period 2000 -o writing_period 100 -o epsilon 1 -o annealing False