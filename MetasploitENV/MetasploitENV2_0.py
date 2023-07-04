import pandas as pd
from util import *

STATE_DICT_DEFINE = {
    'Platform': -1,
    'Number of peers': -1,
    'Peers index': -1,
    'Peers vulnerable port': -1,
    'Windows UAC': -1,
    'Windows in admin group': -1,
    'Windows is Admin': -1,
    'Windows have SYSTEM': -1,
    'Linux Have Root': -1,
    'Linux Kernel': -1,
    'Gather point': -1,
    "Peers platform": -1
}

SUCCESS = 20
FAIL = -1

NUMBER_OF_GATHER_POINT = 2


class EnvironmentTraining:
    def __init__(self, package_num):
        self.vul_port_list = pd.read_csv("MetasploitENV/vul_port.csv")
        self.action_list = pd.read_csv("MetasploitENV/action_list.csv")
        self.module_result = pd.read_csv(f"MetasploitENV/train_package/package_{package_num}/module_result.csv")
        self.state_base = pd.read_csv(f"MetasploitENV/train_package/package_{package_num}/input_state.csv")
        self.peers_info = pd.read_csv(f"MetasploitENV/train_package/package_{package_num}/peers_info.csv")
        self.result_to_stop = pd.read_csv(f"MetasploitENV/train_package/package_{package_num}/result_to_stop.csv")
        self.state_dict = STATE_DICT_DEFINE
        self.state = [0] * (len(self.state_dict))
        self.used_action = [False] * self.action_list.shape[0]

    def setup_test_package(self, pkg_num):
        PrintInfo(f"Setup test package {pkg_num}")
        self.vul_port_list = pd.read_csv("MetasploitENV/vul_port.csv")
        self.action_list = pd.read_csv("MetasploitENV/action_list.csv")
        self.module_result = pd.read_csv(f"MetasploitENV/test_package/package_{pkg_num}/module_result.csv")
        self.state_base = pd.read_csv(f"MetasploitENV/test_package/package_{pkg_num}/input_state.csv")
        self.peers_info = pd.read_csv(f"MetasploitENV/test_package/package_{pkg_num}/peers_info.csv")
        self.result_to_stop = pd.read_csv(f"MetasploitENV/test_package/package_{pkg_num}/result_to_stop.csv")
        self.state_dict = STATE_DICT_DEFINE
        self.state = [0] * (len(self.state_dict))
        self.used_action = [False] * self.action_list.shape[0]

    def setup(self):
        # Setup state
        for index, row in self.state_base.iterrows():
            STATE_DICT_DEFINE['Platform'] = 0 if "win" in row[0] else 1
            STATE_DICT_DEFINE['Number of peers'] = int(row[1])
            STATE_DICT_DEFINE['Peers index'] = int(row[2])
            STATE_DICT_DEFINE['Peers vulnerable port'] = row[3] if row[3] != -1 else 0
            STATE_DICT_DEFINE['Windows UAC'] = int(row[4])
            STATE_DICT_DEFINE['Windows in admin group'] = int(row[5])
            STATE_DICT_DEFINE['Windows is Admin'] = int(row[6])
            STATE_DICT_DEFINE['Windows have SYSTEM'] = int(row[7])
            STATE_DICT_DEFINE['Linux Have Root'] = int(row[8])
            STATE_DICT_DEFINE['Linux Kernel'] = int(row[9])
            STATE_DICT_DEFINE['Gather point'] = int(row[10])
            STATE_DICT_DEFINE['Peers platform'] = int(row[11])
        self.state_dict = STATE_DICT_DEFINE.copy()
        self.update_state_dict_to_state()

    def update_state_dict_to_state(self):
        index = 0
        for key in self.state_dict:
            self.state[index] = self.state_dict[key]
            index += 1

    def check_done(self):
        columns = self.result_to_stop.columns
        total = len(columns)
        now = 0
        for column in columns:
            if self.result_to_stop.loc[0, column] == self.state_dict[column]:
                now += 1
        if now == total:
            return True
        else:
            return False

    def reset(self):
        self.state_dict = STATE_DICT_DEFINE.copy()
        self.update_state_dict_to_state()
        self.used_action = [False] * self.action_list.shape[0]
        return self.state

    def step(self, action_th):
        action_chosen = self.action_list.loc[action_th, "Action List"]
        type_module = self.action_list.loc[action_th, "Type Module"]
        port_target = self.action_list.loc[action_th, "Port target"]
        module_result = self.module_result.loc[action_th, "module result"]
        module_need_root = self.module_result.loc[action_th, "module need root"]
        reward = 0
        PrintInfo(action_chosen)
        if type_module == "privilege escalation":
            if self.state_dict['Platform'] == 0 and self.state_dict[
                "Windows have SYSTEM"] == 0 and module_result == 1 and "windows" in action_chosen:
                self.state_dict['Windows have SYSTEM'] = 1
                self.update_state_dict_to_state()
                reward = SUCCESS
                PrintSuccessiveAction()

            elif self.state_dict['Platform'] == 1 and self.state_dict[
                "Linux Have Root"] == 0 and module_result == 1 and "linux" in action_chosen:
                self.state_dict['Linux Have Root'] = 1
                self.update_state_dict_to_state()
                reward = SUCCESS
                PrintSuccessiveAction()

            else:
                reward = FAIL

        elif type_module == "gather information":
            if self.state_dict["Windows have SYSTEM"] == 1 and "windows" in action_chosen and module_need_root == 1 and \
                    self.state_dict["Gather point"] + 1 <= NUMBER_OF_GATHER_POINT and self.used_action[
                action_th] == False:
                self.state_dict["Gather point"] += 1
                self.update_state_dict_to_state()
                self.used_action[action_th] = True
                reward = SUCCESS
                PrintSuccessiveAction()

            elif "windows" in action_chosen and self.state_dict['Platform'] == 0 and module_need_root == 0 and \
                    self.state_dict["Gather point"] + 1 <= NUMBER_OF_GATHER_POINT and self.used_action[
                action_th] == False:
                self.state_dict["Gather point"] += 1
                self.update_state_dict_to_state()
                self.used_action[action_th] = True
                reward = SUCCESS
                PrintSuccessiveAction()

            elif self.state_dict['Linux Have Root'] == 1 and "linux" in action_chosen and module_need_root == 1 and \
                    self.state_dict["Gather point"] + 1 <= NUMBER_OF_GATHER_POINT and self.used_action[
                action_th] == False:
                self.state_dict["Gather point"] += 1
                self.update_state_dict_to_state()
                self.used_action[action_th] = True
                reward = SUCCESS
                PrintSuccessiveAction()

            elif "linux" in action_chosen and self.state_dict['Platform'] == 1 and module_need_root == 0 and \
                    self.state_dict["Gather point"] + 1 <= NUMBER_OF_GATHER_POINT and self.used_action[
                action_th] == False:
                self.state_dict["Gather point"] += 1
                self.update_state_dict_to_state()
                reward = SUCCESS
                PrintSuccessiveAction()
            else:
                reward = FAIL

        elif type_module == "exploit peers":
            if self.state_dict['Windows have SYSTEM'] == 0 and self.state_dict['Linux Have Root'] == 0:
                reward = FAIL

            if self.state_dict['Peers index'] == self.state_dict["Number of peers"]:
                reward = FAIL

            elif self.state_dict['Peers platform'] == 0 and ("linux" in action_chosen or "multi" in action_chosen):
                reward = FAIL

            elif self.state_dict['Peers platform'] == 1 and 'win' in action_chosen:
                reward = FAIL

            elif "windows" in action_chosen and port_target != 0 and self.peers_info.loc[
                self.state_dict['Peers index'], "vul port"] == port_target and self.state_dict[
                'Platform'] == 0 and module_result == 1 and self.state_dict['Peers index'] + 1 <= self.state_dict[
                "Number of peers"]:
                self.state_dict['Peers index'] += 1

                if self.state_dict['Peers index'] < self.state_dict["Number of peers"]:
                    self.state_dict['Peers platform'] = 0 if "win" in self.peers_info.loc[
                        self.state_dict['Peers index'], "platform"] else 1
                    self.state_dict['Peers vulnerable port'] = 1 if self.peers_info.loc[
                                                                        self.state_dict[
                                                                            'Peers index'], "vul port"] != -1 else 0

                else:
                    self.state_dict['Peers platform'] = -1

                self.update_state_dict_to_state()
                reward = SUCCESS
                PrintSuccessiveAction()

            elif ("linux" in action_chosen or "multi" in action_chosen) and port_target != 0 and self.peers_info.loc[
                self.state_dict['Peers index'], "vul port"] == port_target and self.state_dict[
                'Platform'] == 1 and module_result == 1 and self.state_dict['Peers index'] + 1 <= self.state_dict[
                "Number of peers"]:
                self.state_dict['Peers index'] += 1

                if self.state_dict['Peers index'] < self.state_dict["Number of peers"]:
                    self.state_dict['Peers platform'] = 0 if "win" in self.peers_info.loc[
                        self.state_dict['Peers index'], "platform"] else 1
                    self.state_dict['Peers vulnerable port'] = 1 if self.peers_info.loc[
                                                                        self.state_dict[
                                                                            'Peers index'], "vul port"] != -1 else 0
                else:
                    self.state_dict['Peers platform'] = -1

                self.update_state_dict_to_state()
                reward = SUCCESS
                PrintSuccessiveAction()

            elif module_result == 1 and self.state_dict['Peers index'] + 1 <= self.state_dict["Number of peers"] and \
                    self.peers_info.loc[
                        self.state_dict['Peers index'], "vul port"] == port_target:
                self.state_dict['Peers index'] += 1

                if self.state_dict['Peers index'] < self.state_dict["Number of peers"]:
                    self.state_dict['Peers platform'] = 0 if "win" in self.peers_info.loc[
                        self.state_dict['Peers index'], "platform"] else 1
                    self.state_dict['Peers vulnerable port'] = 1 if self.peers_info.loc[
                                                                        self.state_dict[
                                                                            'Peers index'], "vul port"] != -1 else 0
                else:
                    self.state_dict['Peers platform'] = -1

                self.update_state_dict_to_state()
                reward = SUCCESS
                PrintSuccessiveAction()

            elif self.state_dict['Peers index'] + 1 <= self.state_dict["Number of peers"]:
                self.state_dict['Peers index'] += 1
                print("Siuuuuuu: ", self.state_dict['Peers index'], self.state_dict["Number of peers"])
                if self.state_dict['Peers index'] < self.state_dict["Number of peers"]:
                    self.state_dict['Peers platform'] = 0 if "win" in self.peers_info.loc[
                        self.state_dict['Peers index'], "platform"] else 1
                    self.state_dict['Peers vulnerable port'] = 1 if self.peers_info.loc[
                                                                        self.state_dict[
                                                                            'Peers index'], "vul port"] != -1 else 0
                else:
                    self.state_dict['Peers platform'] = -1

                self.update_state_dict_to_state()
                reward = FAIL

        elif type_module == "skip host":
            if self.state_dict['Peers index'] + 1 <= self.state_dict['Number of peers'] and self.state_dict[
                "Peers vulnerable port"] == 0:
                self.state_dict['Peers index'] += 1
                if self.state_dict['Peers index'] < self.state_dict["Number of peers"]:
                    self.state_dict['Peers platform'] = 0 if "win" in self.peers_info.loc[
                        self.state_dict['Peers index'], "platform"] else 1
                    self.state_dict['Peers vulnerable port'] = 1 if self.peers_info.loc[
                                                                        self.state_dict[
                                                                            'Peers index'], "vul port"] != -1 else 0
                else:
                    self.state_dict['Peers platform'] = -1
                self.update_state_dict_to_state()
                reward = SUCCESS
                PrintSuccessiveAction()

            elif self.state_dict['Peers index'] + 1 <= self.state_dict['Number of peers'] and self.state_dict[
                "Peers vulnerable port"] == 0:
                self.state_dict['Peers index'] += 1
                if self.state_dict['Peers index'] < self.state_dict["Number of peers"]:
                    self.state_dict['Peers platform'] = 0 if "win" in self.peers_info.loc[
                        self.state_dict['Peers index'], "platform"] else 1
                    self.state_dict['Peers vulnerable port'] = 1 if self.peers_info.loc[
                                                                        self.state_dict[
                                                                            'Peers index'], "vul port"] != -1 else 0
                else:
                    self.state_dict['Peers platform'] = -1
                self.update_state_dict_to_state()
                reward = FAIL

            else:
                reward = FAIL

        self.used_action[action_th] = True

        done = self.check_done()
        print(self.state)

        return reward, self.state, done
