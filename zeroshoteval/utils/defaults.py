import os
from argparse import Namespace

from fvcore.common.config import CfgNode
from fvcore.common.file_io import PathManager

from zeroshoteval.utils.collect_env import collect_env_info
from zeroshoteval.utils.logger import setup_logger


def default_setup(cfg: CfgNode, args: Namespace) -> None:
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the ZeroShotEval logger
    2. Log basic information about environment, command line args and config
    3. Save config to the file

    Args:
        args: Command line arguments.
        cfg: Full configuration of a single experiment to use.
    """
    output_dir = cfg.OUTPUT_DIR
    PathManager.mkdirs(output_dir)

    logger = setup_logger(output_dir)

    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    config_file_name: str = "full_config_footprint.yaml"
    if output_dir:
        path = os.path.join(output_dir, config_file_name)
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))
