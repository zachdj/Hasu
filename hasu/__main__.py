"""
Entry point.  CLI to run training and testing scripts
"""

import argparse
from hasu.train import main as training_routine
from hasu.run import main as testing_routine
from hasu.networks.AtariNet import AtariNet


class network_class_parser(argparse.Action):
    def __call__(self, parser, args, network_name, option_string=None):
        network = None
        if network_name == 'atari':
            network = AtariNet
        else:
            print("ERROR: Network %s is not currently supported" % network_name)
        setattr(args, self.dest, network)


def info(args):
    """
    Print system info
    """
    import sys
    print('Python version:')
    print(sys.version)


def main():
    parser = argparse.ArgumentParser(
        description='Hasu: Reinforcement Learning in Starcraft',
        argument_default=argparse.SUPPRESS,
    )
    subcommands = parser.add_subparsers()

    # hasu info
    cmd = subcommands.add_parser('info', description='print system info')
    cmd.set_defaults(func=info)

    # hasu train <args>
    train = subcommands.add_parser('train', description='Train an agent using the A2C algorithm', argument_default=argparse.SUPPRESS)
    train.add_argument('--num_envs', default=32, type=int,
                       help='The number of agents to run during training [DEFAULT: 32]')
    train.add_argument('--step_mul', default=8, type=int,
                       help='The number of observations to skip.  This can be used to limit an agent\'s APM to a fair '
                            'level. A value of 20 is roughly equal to 50 apm while 5 is roughly 200 apm. [DEFAULT: 8]')
    train.add_argument('--max_steps', default=5e6, type=float,
                       help='Maximum number of steps to run during training [DEFAULT: 5 million]')
    train.add_argument('--rollout_steps', default=16, type=int,
                       help='Number of steps to consider for each agent in-between network updates [DEFAULT: 16]')
    train.add_argument('--checkpoint_interval', default=50000, type=int,
                       help='The network will be saved every <checkpoint_interval> steps [DEFAULT: 50000]')
    train.add_argument('--output_directory', default='./output/checkpoints',
                       help='Directory where network checkpoints will be saved [DEFAULT: ./output/checkpoints]')
    train.add_argument('--network_class', default=AtariNet, choices=['atari', 'fully_conv'], action=network_class_parser,
                       help='The type of network to use during training [DEFAULT: atari]')
    train.add_argument('--screen_resolution', default=84, type=int,
                       help='Resolution at which screen observations will be received [DEFAULT: 84]')
    train.add_argument('--minimap_resolution', default=64, type=int,
                       help='Resolution at which minimap observations will be received [DEFAULT: 64]')
    train.add_argument('--use_gpu', action='store_true',
                       help='If set, training will be done on the GPU')
    train.add_argument('--visualize', action='store_true',
                       help='If set, each training environment will be rendered.')
    train.add_argument('--gamma', default=0.99, type=float,
                       help='Discount factor for future rewards [DEFAULT: 0.99]')
    train.add_argument('--value_loss_weight', default=0.5, type=float,
                       help='Weight of value loss in the computation of loss [DEFAULT: 0.5]')
    train.add_argument('--entropy_weight', default=1e-3, type=float,
                       help='Weight of entropy loss in the computation of loss [DEFAULT: 1/1000]')
    train.add_argument('--learning_rate', default=7e-4, type=float,
                       help='Learning rate of neural net [DEFAULT: 7e-4]')
    train.add_argument('--grad_norm_limit', default=40, type=float,
                       help='Clip weight updates to have this norm (helps prevent exploding gradients) [DEFAULT: 40]')
    # TODO: allow specification of observation and action space from command line
    train.set_defaults(func=training_routine)

    # Each subcommand gives an `args.func`.
    # Call that function and pass the rest of `args` as kwargs.
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args = vars(args)
        func = args.pop('func')
        func(**args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
