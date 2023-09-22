import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Test dlc creation')
    parser.add_argument('--model', type=str, help='specify model')
    parser.add_argument('--task', type=str, help='specify task')
    parser.add_argument('--project', type=str, help='specify project')
    parser.add_argument('--iteration', type=str, help='specify iteration')
    parser.add_argument('--snapshot', type=str, help='specify snapshot')
    parser.add_argument('--shuffle', type=str, help='specify shuffle')
    parser.add_argument('--scorer', type=str, default='unknown', help='specify scorer name')
    parser.add_argument('--dataset', type=str, help='specify dataset')
    parser.add_argument('--mm_dataset', type=str, help='specify mmaction2 dataset')
    parser.add_argument('--species', nargs='+', help='specify species')
    parser.add_argument('--all_keypoints', nargs='+', help='specify all keypoints')
    parser.add_argument('--keypoints', nargs='+', help='specify keypoints')
    parser.add_argument('--visibility', type=str, help='specify visibility')
    parser.add_argument('--network', type=str, help='specify network type')
    parser.add_argument('--gpu', type=str, help='specify GPU id')
    parser.add_argument('--cross_validation', type=str, help='specify cross validation')
    args = parser.parse_args()

    return args


def read_config():
    with open('conf.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


def use_deeplabcut(args, cfg):
    from tools.dlc_utils import DLC
    dl = DLC(args)

    if args.task == 'create':
        dl.create_project()
        dl.generate_csv()
    elif args.task == 'create_dataset':
        dl.create_training_dataset()
    elif args.task == 'train':
        dl.train(cfg)
    elif args.task == 'evaluate':
        dl.evaluate()


def use_posec3d(args, cfg):
    if args.task == 'create_dataset':
        from tools.mmaction_utils import MMA
        mm = MMA(args)
        mm.analyze_videos()
        mm.create_training_dataset()
    elif args.task == 'train':
        from tools.mm_distr_train import distributed_training
        distributed_training(args, cfg)


def main():
    args = parse_args()
    cfg = read_config()

    if args.model == 'DEEPLABCUT':
        use_deeplabcut(args, cfg)
    elif args.model == 'POSEC3D':
        use_posec3d(args, cfg)


if __name__ == "__main__":
    main()
