"""Argument parser functions."""

import sys
from argparse import ArgumentParser, REMAINDER, Namespace

from fvcore.common.config import CfgNode

from zeroshoteval.config.defaults import get_cfg_defaults


def parse_args() -> Namespace:
    """
    Parse command line arguments. Two options are available:

        1. Pass `--cfg` argument with a path to YAML config file of a single
        experiment to override specific default experiment settings. Global
        defaults can be found in `zeroshoteval.config.defaults.py`

        2. Pass a list of specific experiment configs to override a default
        ones. This is the same action as `--cfg` passing, but it does not
        require YAML config file creation.
    """
    parser = ArgumentParser(description="ZeroShotEval single experiment launcher.")

    parser.add_argument("--cfg", dest="cfg_file", default=None, type=str,
                        help="Path to YAML config file of a single experiment to override specific default experiment "
                             "settings.")
    parser.add_argument("options", default=None, nargs=REMAINDER,
                        help="A list of specific experiment configs to override a default ones. This is the same action"
                             " as `--cfg` passing, but it does not require YAML config file creation.")

    if len(sys.argv) == 1:
        parser.print_help()

    return parser.parse_args()


def load_config(args: Namespace) -> CfgNode:
    """
    Given the arguments, load and initialize the configs.
    """
    # Setup cfg
    cfg: CfgNode = get_cfg_defaults()

    # Load config from cfg
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    # Load config from command line, overwrite config from opts
    if args.options is not None:
        cfg.merge_from_list(args.options)

    return cfg
