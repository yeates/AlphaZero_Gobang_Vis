# -*- coding:utf-8 -*-
"""
和真人进行博弈
"""

import os, sys
_PATH_ = os.path.dirname(os.path.dirname(__file__))
if _PATH_ not in sys.path:
    sys.path.append(_PATH_)


from logging import getLogger
from multiprocessing import Manager
from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import GoBangEnv, Winner, pretty_print_panel
from chess_zero.lib.model_helper import load_best_model_weight



# logger = getLogger(__name__)


def start(config=Config(config_type='mini')):
    return ManEvaluateWorker(config).start()


class ManEvaluateWorker:
    """
    Worker which evaluates trained models and keeps track of the best one

    Attributes:
        :ivar Config config: config to use for evaluation
        :ivar PlayConfig config: PlayConfig to use to determine how to play, taken from config.eval.play_config
        :ivar ChessModel current_model: currently chosen best model
        :ivar Manager m: multiprocessing manager
        :ivar list(Connection) cur_pipes: pipes on which the current best ChessModel is listening which will be used to
            make predictions while playing a game.
    """
    def __init__(self, config: Config):
        """
        :param config: Config to use to control how evaluation should work
        """
        self.config = config
        self.play_config = config.eval.play_config
        self.current_model = self.load_current_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.current_model.get_pipes(self.play_config.search_threads) for _ in range(self.play_config.max_processes)])

    def start(self):
        """
        Start evaluation, endlessly loading the latest models from the directory which stores them and
        checking if they do better than the current model, saving the result in self.current_model
        """
        while True:
            self.evaluate_model()

    def evaluate_model(self):
        """
        Given a model, evaluates it by playing a bunch of games against the current model.

        :return: true iff this model is better than the current_model
        """
        results = []
        draw_num, win_num, lose_num = 0, 0, 0
        for game_idx in range(0, self.config.eval.game_num):
            score = play_game(config=self.config, cur=self.cur_pipes, robot_white=(game_idx % 2) == 0)
            results.append(score)
            if score == 0.5:
                draw_num += 1
            elif score == 1:
                win_num += 1
            elif score == 0:
                lose_num += 1
            win_rate = sum(results) / len(results)
            game_idx = len(results)
            print(f"第{game_idx}轮游戏: "
                         f"目前人类胜率:{win_rate*100:5.1f}% "
                         f"人类胜利次数:{win_num}, 人类失败次数:{lose_num}, 平局次数:{draw_num}. "
                         )

    def load_current_model(self):
        """
        Loads the best model from the standard directory.
        :return ChessModel: the model
        """
        model = ChessModel(self.config)
        load_best_model_weight(model)
        return model


def play_game(config: Config, cur, robot_white: int) -> (float, GoBangEnv, int):
    """
    Plays a game against models cur and ng and reports the results.

    :param Config config: config for how to play the game
    :param ChessModel cur: should be the current model
    :param ChessModel ng: should be the next generation model
    :param bool ng_no: ng所在的位置,0-1
    :return (float, ChessEnv, bool): the score for the ng model
        (0 for loss, .5 for draw, 1 for win), the env after the game is finished, and a bool
        which is true iff cur played as white in that game.
    """
    cur_pipes = cur.pop()
    env = GoBangEnv().reset()

    configs = config.eval.play_config
    # man
    configs.simulation_num_per_move = 1200
    configs.tau_decay_rate = 0
    #
    current_player = ChessPlayer(config, pipes=cur_pipes, play_config=configs)
    
    if robot_white:
        white, black = current_player, None
    else:
        white, black = None, current_player

    print(f"本局游戏人类为{'黑棋' if robot_white else '白棋'}.")

    while not env.done:
        if env.white_to_move and robot_white:
            action = white.action(env)
        elif env.white_to_move == False and robot_white == False:
            action = black.action(env)
        else:
            # 轮到人类
            print('当前局面如下:')
            pretty_print_panel(env.board.panel)
            print()
            action = input("请输入您要放置的棋子位置:")
            while action not in env.board.legal_moves:
                print("输入有误！请重新输入.")
                action = input("请输入您要放置的棋子位置:")
        env.step(action)

    print('本局游戏结束！当前棋面为:')
    pretty_print_panel(env.board.panel)

    if env.winner == Winner.draw:
        man_score = 0.5
    elif env.white_won == robot_white:
        man_score = 0
    else:
        man_score = 1

    cur.append(cur_pipes)
    return man_score


if __name__ == '__main__':
    import chess_zero.lib.tf_util as ttt
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ttt.set_session_config(allow_growth=True)
    start()
