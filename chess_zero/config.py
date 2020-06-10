"""
Everything related to configuration of running this application
"""

import os
import numpy as np


class PlayWithHumanConfig:
    """
    Config for allowing human to play against an agent using uci

    """
    def __init__(self):
        self.simulation_num_per_move = 1200
        self.threads_multiplier = 2
        self.c_puct = 1 # lower  = prefer mean action value
        self.noise_eps = 0
        self.tau_decay_rate = 0  # start deterministic mode
        self.resign_threshold = None

    def update_play_config(self, pc):
        """
        :param PlayConfig pc:
        :return:
        """
        pc.simulation_num_per_move = self.simulation_num_per_move
        pc.search_threads *= self.threads_multiplier
        pc.c_puct = self.c_puct
        pc.noise_eps = self.noise_eps
        pc.tau_decay_rate = self.tau_decay_rate
        pc.resign_threshold = self.resign_threshold
        pc.max_game_length = 999999


class Options:
    new = False


class ResourceConfig:
    """
    Config describing all of the directories and resources needed during running this project
    """
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir())

        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        self.model_best_config_path = os.path.join(self.model_dir, "model_best_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "model_best_weight.h5")

        self.model_best_distributed_ftp_server = "alpha-chess-zero.mygamesonline.org"
        self.model_best_distributed_ftp_user = "2537576_chess"
        self.model_best_distributed_ftp_password = "alpha-chess-zero-2"
        self.model_best_distributed_ftp_remote_path = "/alpha-chess-zero.mygamesonline.org/"

        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.h5"

        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"

        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.next_generation_model_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


# def flipped_uci_labels():
#     """
#     Seems to somehow transform the labels used for describing the universal chess interface format, putting
#     them into a returned list.
#     :return:
#     """
#     def repl(x):
#         return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

#     return [repl(x) for x in create_uci_labels()]





# ------- 棋盘翻转 -------

PANEL_SIZE = 15  # 棋盘宽度

table_90l = np.zeros((15,15)).tolist()
table_90r = np.zeros((15,15)).tolist()
table_180 = np.zeros((15,15)).tolist()
table_m = np.zeros((15,15)).tolist()
table_90l_m = np.zeros((15,15)).tolist()
table_90r_m = np.zeros((15,15)).tolist()
table_180_m = np.zeros((15,15)).tolist()

tables = [table_90l
            ,table_90r
            ,table_180
            ,table_m
            ,table_90l_m
            ,table_90r_m
            ,table_180_m]

def create_uci_labels(): # 改了原代码
    """
    Creates the labels for the universal chess interface into an array and returns them
    :return:所有可以落子的动作的命令，如'14_1_1'，15行1列放置了一枚黑棋
    """
    labels_array = []
    players = [1, -1] # 1为黑棋，-1为白棋
    for row in range(PANEL_SIZE):
        for col in range(PANEL_SIZE):
            labels_array.append(f'{row}_{col}_{players[0]}')
            labels_array.append(f'{row}_{col}_{players[1]}')
    return labels_array


def create_flip_index(labels):    # 用表储存转换信息
    arr = np.arange(PANEL_SIZE * PANEL_SIZE).reshape((PANEL_SIZE, PANEL_SIZE))
    new_arrs = np.array(flip_combinations(arr))
    for k in range(len(new_arrs)):
        a = new_arrs[k]
        t = tables[k]
        for i in range(PANEL_SIZE*PANEL_SIZE):
            new_row, new_col = np.where(a == i)
            new_row, new_col = new_row[0], new_col[0]
            ori_row, ori_col = int(i / PANEL_SIZE), i % PANEL_SIZE
            t[ori_row][ori_col] = [new_row, new_col]
    
    move_lookup = {move: i for move, i in zip(labels, range(int(len(labels))))}
    
    new_indexes = []
    for t in tables:
        idx = []
        for ori_label in labels:
            ori_row, ori_col, no = ori_label.split('_')
            new_row, new_col = t[int(ori_row)][int(ori_col)]
            idx.append(move_lookup[str(new_row) + '_' + str(new_col) + '_' + str(no)])
        new_indexes.append(idx)
    return new_indexes


def flip_index_adaption(policy):
    fliped_policies = []
    for t in Config.flip_tables:
        temp_p = np.array(policy)
        for i in range(len(policy)):
            temp_p[t[i]] = policy[i]
        fliped_policies.append(temp_p.tolist())
    return fliped_policies
        
        
def flip_combinations(arr): # 返回所有翻转、镜像数组的组合
    arr = np.array(arr)
    arr_90l = flip90_left(arr)
    arr_90r = flip90_right(arr)
    arr_180 = flip180(arr)
    arr_m = np.fliplr(arr)
    arr_90l_m = np.fliplr(arr_90l)
    arr_90r_m = np.fliplr(arr_90r)
    arr_180_m = np.fliplr(arr_180)
    new_arrs = np.array([arr_90l, arr_90r, arr_180, arr_m, arr_90l_m, arr_90r_m, arr_180_m])
    return new_arrs.tolist()
    
def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

def flip90_right(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr



class Config:
    """
    Config describing how to run the application

    Attributes (best guess so far):
        :ivar list(str) labels: labels to use for representing the game using UCI
        :ivar int n_lables: number of labels
        :ivar list(str) flipped_labels: some transformation of the labels
        :ivar int unflipped_index: idk
        :ivar Options opts: options to use to configure this config
        :ivar ResourceConfig resources: resources used by this config.
        :ivar ModelConfig mode: config for the model to use
        :ivar PlayConfig play: configuration for the playing of the game
        :ivar PlayDataConfig play_date: configuration for the saved data from playing
        :ivar TrainerConfig trainer: config for how training should go
        :ivar EvaluateConfig eval: config for how evaluation should be done
    """
    labels = create_uci_labels()
    n_labels = int(len(labels))
    flip_tables = create_flip_index(labels)

    def __init__(self, config_type="mini"):
        """

        :param str config_type: one of "mini", "normal", or "distributed", representing the set of
            configs to use for all of the config attributes. Mini is a small version, normal is the
            larger version, and distributed is a version which runs across multiple GPUs it seems
        """
        self.opts = Options()
        self.resource = ResourceConfig()

        if config_type == "mini":
            import chess_zero.configs.mini as c
        elif config_type == "normal":
            import chess_zero.configs.normal as c
        elif config_type == "distributed":
            import chess_zero.configs.distributed as c
        else:
            raise RuntimeError(f"unknown config_type: {config_type}")
        self.model = c.ModelConfig()
        self.play = c.PlayConfig()
        self.play_data = c.PlayDataConfig()
        self.trainer = c.TrainerConfig()
        self.eval = c.EvaluateConfig()
        self.labels = Config.labels
        self.n_labels = Config.n_labels
        #self.flipped_labels = Config.flipped_labels

    @staticmethod
    def flip_moves(moves):
        """

        : 旋转、翻转原始的moves数据[state_key, policy]，并生成对应的moves数据
        :return: fliped moves' data
        """
        state, policy = moves
        fliped_states = flip_combinations(state)
        fliped_policies = flip_index_adaption(policy)
        return [_ for _ in zip(fliped_states, fliped_policies)]


#Config.unflipped_index = [Config.labels.index(x) for x in Config.flipped_labels]




def _project_dir():
    d = os.path.dirname
    return d(d(os.path.abspath(__file__)))


def _data_dir():
    return os.path.join(_project_dir(), "data")

