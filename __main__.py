from argparse import ArgumentParser

from core.train_speech_embedder import main as train_main

if __name__ != '__main__':
    raise RuntimeError()

main_parser = ArgumentParser()
main_parser.add_argument(main_func=main_parser.print_help)
main_subparser = main_parser.add_subparsers()

train_parser = main_subparser.add_parser('train')
train_parser.set_defaults(main_func=train_main)
train_parser.add_argument('--cfg', type=str, default='config/config.yaml', help='config.yaml path')
train_parser.add_argument('--csv', type=str, default='../data/test_results.csv', help='csv path for writing test results')

args = main_parser.parse_args()
args.main_func(args)
