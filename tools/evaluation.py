"""Main script-launcher for evaliation of ZSL models
"""
from zeroshoteval.evaluation.classification import classification_procedure

import argparse
from config import config

# from src.evaluation_procedures.verification import
# from src.evaluation_procedures.cluster_measurement import


def init_arguments():
    """Initialize arguments
    """
    parser = argparse.ArgumentParser(
        description="Main script-launcher for evaluation of ZSL models"
    )

    parser.add_argument(
        "--procedures",
        required=True,
        help='Comma-separated list of procedures to run for evaluation \
                            (e.g. "classification,cluster_measurement").',
    )

    parser.add_argument(
        "--zsl-embeddings-path",
        required=True,
        help="Path to embeddings computed using ZSL models to be evaluated.",
    )
    parser.add_argument(
        "--zsl-train_embeddings-path",
        default="",
        help="Path to embeddings of training set to use for classification \
                            testing procedure.",
    )
    return parser


def check_arguments(args):
    """Check arguments compatibility and correctness.
    """
    procedures = args.procedures.split(",")
    if "classification" in procedures:
        if args.zsl_train_embeddings_path == "":
            raise ValueError(
                "For classification testing procedure please specify training embeddings."
            )


def load_arguments():
    """Initialize, check and pass arguments.
    """
    parser = init_arguments()
    args = parser.parse_args()
    check_arguments(args)
    return args


def main():
    args = load_arguments()
    procedures = args.procedures.split(",")

    for procedure in procedures:
        if procedure == "classification":
            accuracy = classification_procedure()

        elif procedure == "verification":
            pass

        elif procedure == "cluster_measurement":
            pass


if __name__ == "__main__":
    main()
