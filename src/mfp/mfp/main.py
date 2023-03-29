import logging

from mfp.args import TrainArgs

logger = logging.getLogger(__name__)


def main():
    args = TrainArgs().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger.info(args)

    from mfp.train import train

    train(args)


if __name__ == "__main__":
    main()
