from dataclasses import dataclass, field
from typing import Iterator, Tuple, Union, List, Optional, Dict, Any
import argparse
from collections import OrderedDict

@dataclass
class ArgFields:
    name: Optional[str] = None
    opts: Optional[Dict[str, Any]] = None

@dataclass
class TrainParams:
    model_loc: Optional[str] = None
    save_folder: str = "temp/models"
    n_games: int = 1500
    data_location: str = "temp/games/"
    n_iters: int = -1
    loss_log: int = 5
    train_batchsize: int = 1024
    lr: float = 0.001
    wd: float = 0.0005
    momentum: float = 0.9

    def __setattr__(self, attr, value):
        if value is None:
            value = getattr(self, attr)
        super().__setattr__(attr, value)

    @classmethod
    def add_to_parser(cls, parser):
        for arg_name, arg_field in cls.arg_fields():
            parser.add_argument(arg_field.name, **arg_field.opts)

    @classmethod
    def arg_fields(cls):
        params = OrderedDict(
            model_loc=ArgFields(opts=dict(type=str, help=f"Location of the model to train")),
            save_folder=ArgFields(
                opts=dict(type=str, default=cls.save_folder, help=f"Location to save the model (and checkpoints)")
            ),
            n_iters=ArgFields(
                opts=dict(
                    type=int,
                    default=cls.n_iters,
                    help="Number of batches to train. If -1, it trains indefinitely"
                )
            ),
            loss_log=ArgFields(opts=dict(type=int, default=cls.loss_log, help=f"Log loss every n iterations")),
            train_batchsize=ArgFields(opts=dict(type=int, default=cls.train_batchsize, help=f"Batch size")),
            lr=ArgFields(opts=dict(type=float, default=cls.lr, help=f"Learning rate")),
            wd=ArgFields(opts=dict(type=float, default=cls.wd, help=f"Weight decay")),
            momentum=ArgFields(opts=dict(type=float, default=cls.momentum, help=f"Momentum")),
            data_location=ArgFields(opts=dict(type=str, default=cls.data_location, help=f"Location where the data is")),
            n_games=ArgFields(opts=dict(type=int, default=cls.n_games, help=f"Number of games to sample data from")),
        )

        for param, arg_field in params.items():
            if arg_field.name is None:
                arg_field.name = f"--{param}"
            if arg_field.opts is None:
                arg_field.opts = {}
            if "help" not in arg_field.opts:
                arg_field.opts["help"] = ""
            arg_field.opts["help"] += f" (DEFAULT: {getattr(cls(), param)})"
            yield param, arg_field


@dataclass
class MCTSParams:
    cpuct: float = 1.5
    dirichlet_alpha: float = 1
    n_simulations: int = 1600
    temp: float = 1

    parallel: bool = True
    global_batch_size: int = -1
    batch_size: int = 1
    mcts_threads: int = 1
    vloss: float = 1

    def __setattr__(self, attr, value):
        if value is None:
            value = getattr(self, attr)
        super().__setattr__(attr, value)


    @classmethod
    def add_to_parser(cls, parser):
        for arg_name, arg_field in cls.arg_fields():
            parser.add_argument(arg_field.name, **arg_field.opts)
            
    @classmethod
    def arg_fields(cls):
        params = OrderedDict(
            cpuct=ArgFields(opts=dict(type=float, default=cls.cpuct, help=f"Cpuct. Used to balance exploration and exploitation when searching")),
            dirichlet_alpha=ArgFields(opts=dict(type=float, default=cls.dirichlet_alpha, help=f"Parameter to control the noise added to the root node")),
            n_simulations=ArgFields(opts=dict(type=int, default=cls.n_simulations, help=f"Number of simulations to perform")),
            temp=ArgFields(opts=dict(type=float, default=cls.temp, help=f"Temperature parameter")),
            parallel=ArgFields(opts=dict(type=bool, default=cls.temp, help=f"Parallelize MCTS using virtual loss")),
            global_batch_size=ArgFields(opts=dict(type=int, default=cls.global_batch_size, help=f"Batch size used if you batch requests between threads. -1 if you want only want to batch request inside a thread using mcts_batchsize")),
            batch_size=ArgFields(opts=dict(type=int, default=cls.batch_size, help=f"Batch size used inside a search. It is recommended that this batchsize be equal to the number of threads for a search")),
            mcts_threads=ArgFields(opts=dict(type=int, default=cls.mcts_threads, help=f"Number of threads used for a search.")),
            vloss=ArgFields(opts=dict(type=float, default=cls.vloss, help=f"Virtual loss")),
        )

        for param, arg_field in params.items():
            if arg_field.name is None:
                arg_field.name = f"--{param}"
            if arg_field.opts is None:
                arg_field.opts = {}
            if "help" not in arg_field.opts:
                arg_field.opts["help"] = ""
            arg_field.opts["help"] += f" (DEFAULT: {getattr(cls(), param)})"
            yield param, arg_field



def to_args(Dataclass):
    return [f"--{name}={value}" for name,value in vars(Dataclass).items()]

def instanciate_params_from_args(Dataclass, args):
    return Dataclass(
        **{param: getattr(args, param, None) for param, _ in Dataclass.arg_fields()}
    )