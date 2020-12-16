#!/usr/bin/env python3

"""Argument parser functions."""
from fvcore.common.config import CfgNode
from zeroshoteval.config.defaults import get_cfg

from argparse import ArgumentParser, REMAINDER, Namespace
import sys


def parse_args() -> Namespace:
    """
    Parse the following arguments for a default parser for PySlowFast users.

    Args:
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = ArgumentParser(
        description="Provide ZeroShotEval pipeline."
    )

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See zeroshoteval/config/defaults.py for all options",
        default=None,
        nargs=REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()

    return parser.parse_args()


def load_config(args: Namespace) -> CfgNode:
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg: CfgNode = get_cfg()

    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    return cfg
