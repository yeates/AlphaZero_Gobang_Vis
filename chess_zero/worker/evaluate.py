"""
Encapsulates the worker which evaluates newly-trained models and picks the best one
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from multiprocessing import Manager
from time import sleep
from datetime import datetime
from threading import Thread
from collections import Counter

from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import GoBangEnv, Winner, pretty_print_panel
from chess_zero.lib.data_helper import get_next_generation_model_dirs, write_game_data_to_file
from chess_zero.lib.model_helper import save_as_best_model, load_best_model_weight, reload_best_model_weight_if_changed

logger = getLogger(__name__)


def start(config: Config):
    return EvaluateWorker(config).start()

class EvaluateWorker:
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
        self.evaluated_model_name = []
        
    def start(self):
        """
        Start evaluation, endlessly loading the latest models from the directory which stores them and
        checking if they do better than the current model, saving the result in self.current_model
        """
        while True:
            ng_model, model_dir = self.load_next_generation_model()
            logger.debug(f"start evaluate model {model_dir}")
            ng_is_great = self.evaluate_model(ng_model)
            if ng_is_great:
                logger.debug(f"New Model become best model: {model_dir}")
                save_as_best_model(ng_model)
                self.current_model = ng_model
            self.move_model(model_dir)

    def evaluate_model(self, ng_model):
        """
        Given a model, evaluates it by playing a bunch of games against the current model.

        :param ChessModel ng_model: model to evaluate
        :return: true iff this model is better than the current_model
        """
        ng_pipes = self.m.list([ng_model.get_pipes(self.play_config.search_threads) for _ in range(self.play_config.max_processes)])
        self.buffer = []
        futures = []
        with ProcessPoolExecutor(max_workers=self.play_config.max_processes) as executor:
            for game_idx in range(self.config.eval.game_num):
                fut = executor.submit(play_game, self.config, cur=self.cur_pipes, ng=ng_pipes, current_white=(game_idx % 2 == 0))
                futures.append(fut)

            results = []
            ng_as_black = []
            ng_as_white = []
            for fut in as_completed(futures):
                # ng_score := if ng_model win -> 1, lose -> 0, draw -> 0.5
                ng_score, env, current_white, data = fut.result()
                print()
                pretty_print_panel(env.board.panel)
                results.append(ng_score)
                win_rate = sum(results) / len(results)
                game_idx = len(results)
                
                if current_white:
                    ng_as_black.append(ng_score)
                else:
                    ng_as_white.append(ng_score)
                
                logger.debug(f"game {game_idx:3}: after {env.num_halfmoves} steps, ng_score={ng_score:.1f} as {'black' if current_white else 'white'} "
                             f"win_rate={win_rate*100:5.1f}% ")
                
                if len(ng_as_black) != 0 and len(ng_as_white) != 0:
                    logger.debug(
                                 f"ng_model win_rate as black:{sum(ng_as_black)/len(ng_as_black)*100:5.1f}%; "
                                 f"ng_model win_rate as white:{sum(ng_as_white)/len(ng_as_white)*100:5.1f}%"
                                 )

                # colors = ("current_model", "ng_model")
                # if not current_white:
                #     colors = reversed(colors)
                # pretty_print(env, colors)

                self.buffer += data
                
                if len(results)-sum(results) >= self.config.eval.game_num * (1-self.config.eval.replace_rate):
                    self.flush_buffer()     # save play data
                    logger.debug(f"lose count reach {results.count(0)} so give up challenge")
                    return False
                if sum(results) >= self.config.eval.game_num * self.config.eval.replace_rate:
                    self.flush_buffer()  # save play data
                    logger.debug(f"win count reach {results.count(1)} so change best model")
                    return True
            


        win_rate = sum(results) / len(results)
        logger.debug(f"winning rate {win_rate*100:.1f}%")
        return win_rate >= self.config.eval.replace_rate

    def move_model(self, model_dir):
        """
        Moves the newest model to the specified directory

        :param file model_dir: directory where model should be moved
        """
        rc = self.config.resource
        new_dir = os.path.join(rc.next_generation_model_dir, "copies", model_dir)
        os.rename(model_dir, new_dir)

    def load_current_model(self):
        """
        Loads the best model from the standard directory.
        :return ChessModel: the model
        """
        model = ChessModel(self.config)
        load_best_model_weight(model)
        return model

    def load_next_generation_model(self):
        """
        Loads the next generation model from the standard directory
        :return (ChessModel, file): the model and the directory that it was in
        """
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)
            if dirs:
                break
            logger.info("There is no next generation model to evaluate")
            sleep(60)
        model_dir = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        i = -1
        while model_dir in self.evaluated_model_name:
        #while Counter(self.evaluated_model_name)[model_dir] >= 2:   # evaluate a model only once
            i -= 1
            model_dir = dirs[i]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model = ChessModel(self.config)
        model.load(config_path, weight_path)
        self.evaluated_model_name.append(model_dir)
        return model, model_dir

    def flush_buffer(self):
        """
        Flush the play data buffer and write the data to the appropriate location
        """
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        thread = Thread(target=write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []

def play_game(config, cur, ng, current_white: bool) -> (float, GoBangEnv, bool):
    """
    Plays a game against models cur and ng and reports the results.

    :param Config config: config for how to play the game
    :param ChessModel cur: should be the current model
    :param ChessModel ng: should be the next generation model
    :param bool current_white: whether cur should play white or black
    :return (float, GoBangEnv, bool): the score for the ng model
        (0 for loss, .5 for draw, 1 for win), the env after the game is finished, and a bool
        which is true iff cur played as white in that game.
    """
    cur_pipes = cur.pop()
    ng_pipes = ng.pop()
    env = GoBangEnv().reset()

    current_player = ChessPlayer(config, pipes=cur_pipes, play_config=config.eval.play_config)
    ng_player = ChessPlayer(config, pipes=ng_pipes, play_config=config.eval.play_config)
    if current_white:
        white, black = current_player, ng_player
    else:
        white, black = ng_player, current_player

    while not env.done:
        if env.white_to_move:
            action = white.action(env)
        else:
            action = black.action(env)
        env.step(action)

    if env.winner == Winner.draw:
        ng_score = 0.5
    elif env.white_won == current_white:
        ng_score = 0
    else:
        ng_score = 1
    

# ----- 整理moves -----
    if env.winner == Winner.white:
        black_score, white_score = -1, 1
    elif env.winner == Winner.black:
        black_score, white_score = 1, -1
    else:
        black_score, white_score = -0.5, -0.5

    black.finish_game(black_score)
    white.finish_game(white_score)

    data = []
    for i in range(len(black.moves)):
        data.append(black.moves[i])
        if i < len(white.moves):
            data.append(white.moves[i])

# --------------------

    cur.append(cur_pipes)
    ng.append(ng_pipes)
    return ng_score, env, current_white, data
