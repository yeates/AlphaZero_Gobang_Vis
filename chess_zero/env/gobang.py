"""
五子棋核心逻辑
created time: 2020年06月08日
author: 喻永生
"""
import enum 
import numpy as np
import copy 
import time

PANEL_SIZE = 15

class Player(enum.Enum):
    WHITE = -1
    BLACK = 1
    WNAME = '白子'
    BNAME = '黑子'
    
class Direct(enum.Enum):
    NONE = 0    # 未定方向 
    DOWN = 1 
    RIGHT = 2
    TILTR = 3   # 斜右向下
    TILTL = 4   # 斜左向下

class Board:
    def __init__(self):
        self.turn = Player.BLACK.value # 当前回合行动玩家。黑子先下
        # 15*15的棋盘，0为空，1为黑子，-1为白子
        self.panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    
    @property
    def legal_moves(self):
        moves = np.array([])
        legal_positions = zip(np.where(self.panel == 0)[0], np.where(self.panel == 0)[1])
        for row, col in legal_positions:
            moves = np.append(moves, str(row) + '_' + str(col) + '_' + str(self.turn))
        return moves
    
    def push_uci(self, action): # 执行动作
        row, col, no = action.split('_')
        row, col, no = int(row), int(col), int(no)
        if no != Player.WHITE.value and no != Player.BLACK.value:
            print(f"不存在玩家:{no}！")
        else:
            self.panel[row, col] = no
            #print(f"成功在{row+1}行,{col+1}列放置一枚{Player.WNAME.value if no == Player.WHITE.value else Player.BNAME.value}.")
            self.next_round()
    
    def regret(self, action):
        row, col, no = action.split('_')
        row, col, no = int(row), int(col), int(no)
        self.panel[row, col] = 0 # 置为0
        self.next_round()
    
    def result(self):   # 获取当前局面结果
        #starttime = time.time()
        score_b, score_w = 0, 0
        panel = self.panel
        for r in range(panel.shape[0]):
            for c in range(panel.shape[1]):
                score_b = find_five_chess(0, panel, Player.BLACK.value, r, c, Direct.NONE.value)
                score_w = find_five_chess(0, panel, Player.WHITE.value, r, c, Direct.NONE.value)
                if max(score_b, score_w) >= 5: # 大于5个算赢
                    if score_b > score_w: # 黑胜利
                        return '0-1'
                    else:
                        return '1-0'      # 白胜利
        if sum(sum(panel==0)) == 0:       # 棋盘下满了，则平局
            return '1-1'
        #print(f'consume time:{time.time() - starttime}')
        return '*'                        # 以上情况都没有，则没有结束
        

    def next_round(self):  
        self.turn = Player.BLACK.value if self.turn == Player.WHITE.value else Player.WHITE.value

    def copy(self):
        board = copy.copy(self)
        board.panel = copy.deepcopy(self.panel)
        return board

def find_five_chess(num, panel, no=Player.BLACK.value, row=0, col=0, direct=Direct.NONE.value): # 迭代查找棋盘中最长连珠的个数
    if row >= panel.shape[0] or col >= panel.shape[1]:
        return num
    if panel[row, col] == no:
        if direct == Direct.NONE.value:
            down = find_five_chess(num+1, panel, no, row+1, col, Direct.DOWN.value) # 向下搜索
            right = find_five_chess(num+1, panel, no, row, col+1, Direct.RIGHT.value) # 向右搜索
            tiltr = find_five_chess(num+1, panel, no, row+1, col+1, Direct.TILTR.value) # 向右下搜索
            tiltl = find_five_chess(num+1, panel, no, row+1, col-1, Direct.TILTL.value) # 向左下搜索
            return max(down, right, tiltr, tiltl)
        elif direct == Direct.DOWN.value:
            return find_five_chess(
                num+1, panel, no, row+1, col, Direct.DOWN.value)  # 向下搜索
        elif direct == Direct.RIGHT.value:
            return find_five_chess(
                num+1, panel, no, row, col+1, Direct.RIGHT.value)  # 向右搜索
        elif direct == Direct.TILTR.value:
            return find_five_chess(
                num+1, panel, no, row+1, col+1, Direct.TILTR.value)  # 向右下搜索
        elif direct == Direct.TILTL.value:
            return find_five_chess(
                num+1, panel, no, row+1, col-1, Direct.TILTL.value)  # 向左下搜索
    else:
        return num

            
                
