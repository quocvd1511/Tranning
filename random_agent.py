import random
from MetasploitENV import MetasploitENV2_0 as metaEnv


def test(try_quantity=200):
    privilege_escalation = 0
    gather_hashdump = 0
    lateral_movement = 0
    num_lateral_movement = 0
    state = env.reset()
    done = False
    total_reward = 0
    num_of_act = 0
    while not done:
        if num_of_act == try_quantity:
            done = True
            break
        action = random.randint(0, 98)
        reward, next_state, done = env.step(action)
        if reward > 0:
            if env.action_list.loc[action, "Type Module"] == "privilege escalation":
                privilege_escalation = num_of_act + 1
            elif env.action_list.loc[action, "Type Module"] == "gather information":
                gather_hashdump = num_of_act + 1
            else:
                lateral_movement = num_of_act + 1
                num_lateral_movement += 1

        total_reward += reward
        state = next_state
        num_of_act += 1

    print("Test Complete: Total Reward = {}".format(total_reward))
    return privilege_escalation, gather_hashdump, lateral_movement, num_lateral_movement


env = metaEnv.EnvironmentTraining(package_num=1)
state_dim = len(env.state)
action_dim = env.action_list.shape[0]
env.setup()

number_of_test = 100
for pkg_num in range(1, 5):
    pe = 0
    n_pe = 0
    gh = 0
    n_gh = 0
    lm = 0
    n_lm = []
    name_test_log = f"Result/Testing/RandomAgent/random_agent.log"
    for num in range(1, number_of_test + 1):
        env = metaEnv.EnvironmentTraining(package_num=1)
        env.setup_test_package(pkg_num)
        env.setup()
        privilege_escalation, gather_hashdump, lateral_movement, num_lateral_movement = test()
        pe += privilege_escalation
        gh += gather_hashdump
        if num_lateral_movement >= env.peers_info.shape[0]:
            lm += 1
        if privilege_escalation > 0:
            n_pe += 1
        if gather_hashdump > 0:
            n_gh += 1
        n_lm.append(num_lateral_movement)
    print("======================================")
    file_write = open(name_test_log, "a")
    file_write.writelines(f'num of privilege escalation success: {n_pe}/100\n')
    file_write.writelines(f'num of gather hashdump success: {n_gh}/100\n')
    file_write.writelines(f"privilege escalation: {pe / n_pe}\n")
    file_write.writelines(f"gather information: {gh / n_gh}\n")
    file_write.writelines(f"lateral movement successful all: {lm}/100\n")
    file_write.writelines(f"lateral movement compromised each time: {n_lm}\n")
    file_write.writelines("\n\n")
