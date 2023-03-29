import argparse

DATASET_NAMES = ["rico", "crello"]


class BaseArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.parser.add_argument(
            "--dataset_name",
            required=True,
            choices=DATASET_NAMES,
            help="Name of the dataset.",
        )
        self.parser.add_argument(
            "--data_dir",
            # required=True,
            help="The GCS or local path of the data location.",
        )
        self.parser.add_argument(
            "--weights",
            default=None,
            type=str,
            help="Path to the initial model weight.",
        )
        self.parser.add_argument(
            "--latent_dim",
            default=256,
            type=int,
            help="Latent dimension.",
        )
        self.parser.add_argument(
            "--num_blocks",
            default=4,
            type=int,
            help="Number of stacked blocks in sequence encoder.",
        )
        self.parser.add_argument(
            "--arch_type",
            default="oneshot",
            help="Overall model type",
        )
        self.parser.add_argument(
            "--block_type",
            default="deepsvg",
            help="Stacked block type.",
        )
        self.parser.add_argument(
            "--l2",
            default=1e-2,
            type=float,
            help="Scalar coefficient for L2 regularization.",
        )
        self.parser.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            help="Scalar ratio for dropout in transformer",
        )
        self.parser.add_argument(
            "--masking_method",
            type=str,
            default="random",
        )
        self.parser.add_argument(
            "--seq_type",
            type=str,
            default="default",
            choices=["default", "flat", "concat_enc"],
            help="transformer's input is: element-wise feature (default), field-wise feature (flat)",
        )
        self.parser.add_argument("--log_level", default="INFO", type=str)
        self.parser.add_argument("--verbose", default=2, type=int)
        self.parser.add_argument("--seed", default=0, type=int)
        self.parser.add_argument("--mult", default=1.0, type=float)
        self.parser.add_argument(
            "--context",
            default=None,
        )
        self.parser.add_argument(
            "--input_dtype",
            type=str,
            default="set",
            choices=["set", "shuffled_set"],
        )
        self.parser.add_argument("--batch_size", default=256, type=int)

    def parse_args(self):
        return self.parser.parse_args()


class TrainArgs(BaseArgs):
    def __init__(self):
        super().__init__()
        self.parser.add_argument(
            "--job-dir",
            required=True,
            help="The GCS or local path of logs and saved models.",
        )
        self.parser.add_argument(
            "--num_epochs",
            default=500,
            type=int,
            help="Number of epochs to train.",
        )
        self.parser.add_argument(
            "--learning_rate",
            default=1e-4,
            type=float,
            help="Base learning rate.",
        )
        self.parser.add_argument(
            "--enable_profile",
            dest="enable_profile",
            action="store_true",
            help="Enable profiling for tensorboard. (See tensorflow/tensorboard#3149)",
        )
        self.parser.add_argument(
            "--validation_freq",
            default=10,
            type=int,
            help="Validation frequency in terms of epochs.",
        )

    def __call__(self):
        return self.parser.parse_args()
