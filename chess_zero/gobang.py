import os, sys
_PATH_ = os.path.dirname(os.path.dirname(__file__))
if _PATH_ not in sys.path:
    sys.path.append(_PATH_)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pygame
import traceback
import copy
from pygame.locals import *
from chess_zero.env.chess_env import GoBangEnv, Winner, PANEL_SIZE, Player
from chess_zero.lib.model_helper import load_best_model_weight
from multiprocessing import Manager
from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config

from time import sleep



# ------ chess board settings ------
## color
class Color:
    background = (201,202,187)
    checkerboard = (80,80,80)
    button = (52,53,44)

class PvEWorker:
    def __init__(self, config: Config):
        """
        :param config: Config to use to control how evaluation should work
        """
        # ------ model profile ------
        self.config = config
        self.play_config = config.eval.play_config
        self.current_model = self.load_current_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.current_model.get_pipes(
            self.play_config.search_threads) for _ in range(self.play_config.max_processes)])

    def start(self):
        game_idx = 0
        while True:
            cur = self.cur_pipes.pop()
            play_config = self.play_config
            play_config.simulation_num_per_move = 100
            play_config.tau_decay_rate = 0
            robot = ChessPlayer(self.config, pipes=cur, play_config=play_config)
            score = play_game(robot, (game_idx % 2) == 0)
            game_idx += 1
            self.cur_pipes.append(cur)

    def load_current_model(self):
        """
        Loads the best model from the standard directory.
        :return ChessModel: the model
        """
        model = ChessModel(self.config)
        load_best_model_weight(model)
        return model

#绘制棋盘
def draw_board(screen):
    #填充背景色
    screen.fill(Color.background)
    #Background=pygame.image.load("background.jpg").convert_alpha()
    #screen.blit(Background,(0,0))
    #画棋盘
    for i in range(21):
        pygame.draw.line(screen, Color.checkerboard, (40*i+3, 3), (40*i+3, 803))
        pygame.draw.line(screen, Color.checkerboard, (3, 40*i+3), (803, 40*i+3))
    #画边线
    pygame.draw.line(screen, Color.checkerboard, (3, 3), (803, 3),5)
    pygame.draw.line(screen, Color.checkerboard, (3, 3), (3, 803),5)
    pygame.draw.line(screen, Color.checkerboard, (803, 3), (803, 803),5)
    pygame.draw.line(screen, Color.checkerboard, (3, 803), (803, 803),5)

    #画定位点
    pygame.draw.circle(screen, Color.checkerboard, (163, 163), 6)
    pygame.draw.circle(screen, Color.checkerboard, (163, 643), 6)
    pygame.draw.circle(screen, Color.checkerboard, (643, 163), 6)
    pygame.draw.circle(screen, Color.checkerboard, (643, 643), 6)
    pygame.draw.circle(screen, Color.checkerboard, (403, 403), 6)

    #画‘悔棋’‘重新开始’跟‘退出’按钮
    pygame.draw.rect(screen, Color.button,[900,350,120,100],5)
    pygame.draw.rect(screen, Color.button,[900,500,200,100],5)
    pygame.draw.rect(screen, Color.button,[900,650,200,100],5)
    s_font = pygame.font.Font('chess_zero/gui/STFANGSO.ttf', 40)
    text1=s_font.render("悔棋",True,Color.button)
    text2=s_font.render("重新开始",True,Color.button)
    text3=s_font.render("退出游戏",True,Color.button)
    screen.blit(text1,(920,370))
    screen.blit(text2,(920,520))
    screen.blit(text3,(920,670))

#绘制棋子（横坐标，纵坐标，屏幕，棋子颜色（1代表黑，2代表白））
def plot_chess(x, y, screen, color):
    if color==1:
        Black_chess=pygame.image.load("chess_zero/gui/Black_chess.png").convert_alpha()
        screen.blit(Black_chess,(40*x+3-15,40*y+3-15))
    if color==-1:
        White_chess = pygame.image.load("chess_zero/gui/White_chess.png").convert_alpha()
        screen.blit(White_chess,(40*x+3-15,40*y+3-15))

#绘制带有棋子的棋盘
def draw_board_with_chess(map, screen):
    screen.fill(Color.background)
    draw_board(screen)
    for i in range(PANEL_SIZE):
        for j in range(PANEL_SIZE):
            plot_chess(i+1,j+1,screen,map[i][j])

#绘制提示器（类容，屏幕，字大小）
def put_text(text, screen, font_size):
    #先把上一次的类容用一个矩形覆盖
    pygame.draw.rect(screen, Color.background, [850,100,1200,100])
    #定义字体跟大小
    s_font=pygame.font.Font('chess_zero/gui/STFANGSO.ttf',font_size)
    #定义类容，是否抗锯齿，颜色
    s_text=s_font.render(text, True, Color.button)
    #将字放在窗口指定位置
    screen.blit(s_text,(880,100))
    pygame.display.flip()


#主函数
def play_game(robot: ChessPlayer, robot_white: int) -> (float, GoBangEnv, int):
    env = GoBangEnv().reset()

    screen = pygame.display.set_mode([1200,806])    #定义窗口
    pygame.display.set_caption("五子棋")    #定义窗口名字
    put_text(
        f'本局游戏人类为{Player.BNAME if robot_white else Player.WNAME}.', screen, 28)
    #在窗口画出棋盘，提示器以及按钮
    draw_board(screen)
    pygame.display.flip()
    clock=pygame.time.Clock()

    while not env.done: # 一局游戏开始
        if not env.white_to_move:
            no = 1  # 黑子编号为1
            put_text('黑棋落子', screen, 54)
        else:
            no = -1 # 白子编号为-1
            put_text('白棋落子', screen, 54)
        # 判断是否为robot下棋
        if env.white_to_move and robot_white:
            action = robot.action(env)
            print(action)
            i, j, no = action.split('_')
            plot_chess(int(i)+1, int(j)+1, screen, int(no))
            pygame.display.flip()
            env.step(action)
        elif not env.white_to_move and not robot_white:
            action = robot.action(env)
            i, j, no = action.split('_')
            plot_chess(int(i)+1, int(j)+1, screen, int(no))
            pygame.display.flip()
            print(action)
            env.step(action)
        else:
            # 轮到人类
            block = False
            for event in pygame.event.get():
                # 关闭窗口
                if event.type ==pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # 点击窗口里面类容则完成相应指令
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        x, y=event.pos[0], event.pos[1]
                        # 如果点击‘重新开始’
                        if 900< x <1100 and 500< y <600:
                            return
                        #点击‘退出游戏’，退出游戏
                        elif 900<x<1100 and 650<y<750:
                            pygame.quit()
                            sys.exit()
                        #点击‘悔棋’
                        elif 900<x<1020 and 350<y<450 and env.previous_actions.shape[0] >= 2:
                            env.regret_n_steps(step=2)
                            #将map显示出来
                            draw_board_with_chess(env.board.panel, screen)
                            #悔棋完成，阻止再次悔棋
                            x,y=0,0

                        for i in range(PANEL_SIZE):
                            for j in range(PANEL_SIZE):
                                #点击棋盘相应位置
                                if i*40+3+20<x<i*40+3+60 and j*40+3+20<y<j*40+3+60 and not env.board.panel[i, j] and not block:
                                    block = True
                                    #在棋盘相应位置落相应颜色棋子
                                    plot_chess(i+1, j+1, screen, no)
                                    action = f'{i}_{j}_{no}'
                                    print(action)
                                    pygame.display.flip()
                                    env.step(action)
        clock.tick(60)

    if env.white_won:
        put_text('白棋胜利，请重新游戏',screen,30)
    else:
        put_text('黑棋胜利，请重新游戏',screen,30)
    sleep(10)


if __name__ == "__main__":
    try:
        pygame.init()
        pygame.mixer.init()
        import chess_zero.lib.tf_util as tu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        tu.set_session_config(allow_growth=True)
        PvEWorker(Config(config_type='mini')).start()
    except SystemExit:
        pass
    except:
        traceback.print_exc()
        pygame.quit()
        input()

