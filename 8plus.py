import numpy as np
import os
from glob import glob
import json

PANEL_SIZE = 3

table_90l = np.zeros((PANEL_SIZE, PANEL_SIZE)).tolist()
table_90r = np.zeros((PANEL_SIZE, PANEL_SIZE)).tolist()
table_180 = np.zeros((PANEL_SIZE, PANEL_SIZE)).tolist()
table_m = np.zeros((PANEL_SIZE, PANEL_SIZE)).tolist()
table_90l_m = np.zeros((PANEL_SIZE, PANEL_SIZE)).tolist()
table_90r_m = np.zeros((PANEL_SIZE, PANEL_SIZE)).tolist()
table_180_m = np.zeros((PANEL_SIZE, PANEL_SIZE)).tolist()

tables = [table_90l, table_90r, table_180,
          table_m, table_90l_m, table_90r_m, table_180_m]


def create_uci_labels():  # 改了原代码
    """
    Creates the labels for the universal chess interface into an array and returns them
    :return:所有可以落子的动作的命令，如'14_1_1'，15行1列放置了一枚黑棋
    """
    labels_array = []
    players = [1, -1]  # 1为黑棋，-1为白棋
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

    move_lookup = {move: i for move, i in zip(
        labels, range(int(len(labels))))}

    new_indexes = []
    for t in tables:
        idx = []
        for ori_label in labels:
            ori_row, ori_col, no = ori_label.split('_')
            new_row, new_col = t[int(ori_row)][int(ori_col)]
            idx.append(move_lookup[str(new_row) + '_' +
                                   str(new_col) + '_' + str(no)])
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


def flip_combinations(arr):  # 返回所有翻转、镜像数组的组合
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
    labels = create_uci_labels()
    n_labels = int(len(labels))
    flip_tables = create_flip_index(labels)

def read_game_data_from_file(path):
    try:
        with open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        print(e)

def write_game_data_to_file(path, data):
    try:
        with open(path, "wt") as f:
            json.dump(data, f)
    except Exception as e:
        print(e)
        
        
# def testify():
    

if __name__ == '__main__':
    # pattern = '/home/ideal/yuyongsheng/AlphaZero_Gobang_Vis/data/to_data/play_*.json'
    # files = list(sorted(glob(pattern)))
    # for f in files:
    #     print(f)
    #     data = read_game_data_from_file(f)
    #     new_data = []
    #     for state, policy, value in data:
    #         fliped_states = flip_combinations(state)
    #         fliped_policies = flip_index_adaption(policy)
    #         values = [value for _ in range(len(fliped_policies))]
    #         new_data.append([state, policy, value])
    #         for a, b, c in zip(fliped_states, fliped_policies, values):
    #             new_data.append([a, b, c])
    #     write_game_data_to_file(f, new_data)
    
    print(Config.n_labels)
    state, policy = [[[0,1,2],[3,4,5], [6,7,8]], np.arange(Config.n_labels).tolist()]
    fliped_states = flip_combinations(state)
    fliped_policies = flip_index_adaption(policy)
    print(type(fliped_states), type(fliped_policies))
    new_moves = [_ for _ in zip(fliped_states, fliped_policies)]
    print(type(new_moves))
    for state, policy in new_moves:
        print([state, policy])
