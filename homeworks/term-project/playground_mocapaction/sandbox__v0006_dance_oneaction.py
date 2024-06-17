from dm_control import suite
from dm_control import viewer
import numpy as np

env = suite.load(domain_name="humanoid_CMU", task_name="stand")
action_spec = env.action_spec(
)

# Define a uniform random policy.
counter = 0
this_action_done = False
duration = 12
one_counter = 0


def random_policy(time_step):
    global counter
    global this_action_done
    global one_counter

    del time_step

    action = np.zeros(action_spec.shape)

    if this_action_done:
        counter += 1
        this_action_done = False
        if counter == 56:
            counter = 0
    else:
        # if one_counter % 2 == 0:        
        # # if one_counter < 4:
        #     action[counter]= 1.0
        # else:
        #     action[counter]= -1.0
        
        if one_counter < int(duration/4):      
        # if one_counter < 4:
            action[counter]= 1.0
        elif one_counter < int(duration/4)*2:
            action[counter]= -1.0
        elif one_counter < int(duration/4)*3:
            action[counter]= 1.0
        else:
            action[counter]= -1.0

        one_counter += 1
        if one_counter == duration:
            one_counter = 0
            this_action_done = True

    print(f'action num: {counter}, duration: {one_counter}')
    return action

# Launch the viewer application.
viewer.launch(env, policy=random_policy)