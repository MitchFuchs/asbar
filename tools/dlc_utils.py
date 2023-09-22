"""
This file is derived from the following sources:
- https://github.com/shoopshoop/OMC/blob/main/models/deeplabcut/deeplabcut_baseline.py
Modifications added: multianimal, visibility, species, edit_config
"""
import os
import time
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
print('importing dlc')
import deeplabcut as dlc
from tools.datasets import *

#from https://stackoverflow.com/questions/62466877/self-traceback-tf-stack-extract-stack
import tensorflow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#from https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

class DLC:
    def __init__(self, args=None):
        self.db = args.dataset
        self.project = args.project
        if self.project is None:
            self.scorer = args.scorer
            # self.db = args.dataset
            self.species = list(args.species)
            self.all_keypoints = list(args.all_keypoints)
            self.keypoints = list(args.keypoints)
            self.boolean_keypoints = self.get_boolean_keypoints()
            self.only_visible_kpt = self.get_visibility(args.visibility)
            # self.annotation_file = self.get_annotation_file()
            self.img_dir_train = self.get_image_dir_train()
            self.img_dir_val = self.get_image_dir_val()
            self.config, self.cfg, self.csv_name = None, None, None
            self.ma = True  # multi animal project
            self.iteration = 0
            self.n_iterations = 1
        else:
            self.project_dir = self.get_project_dir()
            self.config = self.get_config()
            self.network_type = args.network
            self.iteration = args.iteration
            self.n_iterations = self.count_all_iterations()
            self.cfg = self.edit_iteration_in_config()
            self.iteration_dir = self.get_iteration_dir()
            self.dest_folder = self.get_dest_folder()
            self.vid_dir = self.get_vid_dir()
            self.gpu_id = args.gpu
            if args.shuffle is not None:
                self.shuffle = self.get_shuffle(args.shuffle)
            if args.snapshot is not None:
                self.snapshot = args.snapshot
                self.snapshots = self.get_snapshots()
                self.cfg = self.edit_snapshot_in_config()
            if args.cross_validation is not None:
                self.cross_validation = args.cross_validation


    def create_project(self):
        timestamp = time.strftime("%H%M%S")

        self.config = dlc.create_new_project(project='dlc-' + timestamp,
                                             experimenter=self.scorer,
                                             videos=['./resources/dummy_video.avi'],
                                             working_directory='./models/deeplabcut',
                                             copy_videos=True,
                                             multianimal=True)

        edits = {
            'individuals': ['ind1'],
            'multianimalbodyparts': self.keypoints}
        self.cfg = self.edit_config(edits)
        os.mkdir(os.path.join(self.cfg['project_path'], 'analyzed_videos'))

    def generate_csv(self):
        data = Dataset(self)
        # bodyparts = self.cfg['multianimalbodyparts']

        # img_paths = [os.path.join(self.img_dir, data.imgs[i]) for i in range(len(data.imgs))]
        landmark_coords = [data.landmarks[i] for i in range(len(data.landmarks))]
        visibility = [[val for val in data.visibility[i] for _ in range(2)] for i in range(len(data.visibility))]
        visible_coords = [[a * b for a, b in zip(landmark_coords[i], visibility[i])] for i in
                          range(len(data.visibility))]
        final_coords = [[x if x > 0 else '' for x in visible_coords[i]] for i in range(len(data.visibility))]
        filename = f'CollectedData_{self.scorer}.csv'
        self.csv_name = os.path.join(self.cfg['project_path'], 'labeled-data', 'dummy_video', filename)
        with open(self.csv_name, "w") as f:
            # header info
            header = 'scorer,' + ",".join([self.scorer] * (2 * len(self.keypoints))) + '\n'
            if self.ma:
                header += 'individuals,' + ",".join(['ind1'] * (2 * len(self.keypoints))) + '\n'
            header += 'bodyparts,' + ",".join([bp + "," + bp for bp in self.keypoints]) + '\n'
            header += 'coords,' + ",".join(['x', 'y'] * len(self.keypoints))
            f.write(header + '\n')
            # for i, img_path in enumerate(img_paths):
            for i, img in data.imgs.items():
                # img = img_path[img_path.rfind('/') + 1:]
                f.write('labeled-data/dummy_video/' + img + "," + \
                        ",".join([str(val) for val in final_coords[i]]) + "\n")
                # os.symlink(src=os.path.join(os.getcwd(), self.img_dir, img),
                #            dst=os.path.join(self.cfg['project_path'], 'labeled-data', 'dummy_video',
                #                             img[img.rfind('/') + 1:]),
                #            target_is_directory=True)

                # if i < data.n_train:
                #     img_dir = self.img_dir_train
                # else:
                #     img_dir = self.img_dir_val

                os.symlink(src=os.path.join(os.getcwd(), 'data', self.db, data.split[i], img),
                           dst=os.path.join(self.cfg['project_path'], 'labeled-data', 'dummy_video', img),
                           target_is_directory=True)

        dlc.convertcsv2h5(self.config, False)

    def create_training_dataset(self):
        if self.cross_validation == 'NO':
            dlc.create_multianimaltraining_dataset(
                config=self.config,
                net_type=self.network_type)
        else:
            if self.cross_validation == '5-fold-CV':
                n_splits = 5
            elif self.cross_validation == '10-fold-CV':
                n_splits = 10
            X = np.array(range(self.get_n_labels()))
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
            all_train_indices, all_test_indices = [], []
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                all_train_indices.append(train_index)
                all_test_indices.append(test_index)
            dlc.create_multianimaltraining_dataset(
                config=self.config, num_shuffles=n_splits, trainIndices=all_train_indices, testIndices=all_test_indices,
                net_type=self.network_type)

    def train(self, cfg):
        for shuffle in self.shuffle:
            dlc.train_network(self.config, shuffle=shuffle, gputouse=self.gpu_id,  max_snapshots_to_keep=None,
                              displayiters=cfg['dlc']['displayiters'], saveiters=cfg['dlc']['saveiters'],
                              maxiters=cfg['dlc']['maxiters'], allow_growth=True)

    def evaluate(self):
        dlc.evaluate_network(self.config, Shuffles=self.shuffle, rescale=False, gputouse=self.gpu_id, test_only=True)

    def analyze_videos(self):
        dlc.analyze_videos(self.config, videos=[self.vid_dir],
                           gputouse=self.gpu_id, destfolder=self.dest_folder, allow_growth=True, auto_track=False)

    def edit_config(self, edits):
        dlc.auxiliaryfunctions.edit_config(self.config, edits)
        return dlc.utils.auxiliaryfunctions.read_plainconfig(self.config)

    def edit_iteration_in_config(self):
        if self.iteration == "NEW":
            n_iteration = self.n_iterations
        else:
            n_iteration = int(self.iteration.split('-')[1])
        edits = {'iteration': n_iteration}
        return self.edit_config(edits)

    def edit_snapshot_in_config(self):
        print('self.snapshot', self.snapshot)
        if self.snapshot == "last":
            snapshot_id = -1
        elif self.snapshot == "best_test_error":
            snapshot_id = self.get_best_snapshot()
        else:
            snapshot_id = self.snapshots.index(self.snapshot)
        print('snapshot_id', snapshot_id)
        edits = {'snapshotindex': snapshot_id}
        return self.edit_config(edits)

    def get_snapshots(self):
        print(self.shuffle)
        if self.shuffle[0] == 'all':
            model = os.listdir(self.iteration_dir)[0] # WARNING!! only works for shuffle 1
            print('Warning! Using first shuffle found')
        else:
            model = [x for x in os.listdir(self.iteration_dir) if x.endswith('shuffle'+str(self.shuffle[0]))][0]
        print('model', model)
        snapshots = os.listdir(os.path.join(self.iteration_dir, model, 'train'))
        snapshots = [x.split('.meta')[0] for x in snapshots if x.endswith('.meta')]
        snapshots = sorted(snapshots, key=lambda s: int(re.search(r'\d+', s).group()))
        return snapshots

    def get_best_snapshot(self):
        eval_folder = os.path.join(self.project_dir, 'evaluation-results', 'iteration-' + str(self.cfg['iteration']))
        df = pd.read_csv(os.path.join(eval_folder, 'CombinedEvaluation-results.csv'), index_col=1)
        snapshot_iter = df[[' Test error(px)']].idxmin()[0]
        print(str(snapshot_iter))
        self.snapshot = 'snapshot-' + str(snapshot_iter)
        return self.snapshots.index(self.snapshot)

    def get_shuffle(self, shuffle):
        # if shuffle is None:
        #     shuffle = self.cfg['project_path']
        if shuffle == 'all':
            n_shuffle = len(os.listdir(os.path.join(self.iteration_dir)))
            return list(range(1, n_shuffle+1))
        else:
            shuffle = shuffle.split("shuffle",1)[1]
            return list(shuffle)


    def get_project_dir(self):
        return os.path.join(os.getcwd(), f'models/deeplabcut/{self.project}')

    def get_iteration_dir(self):
        return os.path.join(self.project_dir, 'dlc-models', 'iteration-' + str(self.cfg['iteration']))

    def get_n_labels(self):
        labeled_data = os.path.join(self.project_dir, 'labeled-data', 'dummy_video')
        labeled_data = os.path.join(self.project_dir, 'labeled-data', 'final_preds')
        # print(os.path.isdir(labeled_data))
        csv_file = [os.path.join(labeled_data, x) for x in os.listdir(labeled_data) if x.endswith('csv')][0]
        with open(csv_file, 'r') as f:
            n_labels = len(f.readlines()) - 4 #substract 4 header lines
            # n_labels = len([x for x in os.listdir(labeled_data)]) - 2 #substract 2 CollectedData files
        return n_labels


    def get_dest_folder(self):
        dest_folder = os.path.join(self.project_dir, 'analyzed_videos', 'iteration-'+str(self.cfg['iteration']))
        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)
        return dest_folder

    def get_config(self):
        return os.path.join(self.project_dir, 'config.yaml')

    def count_all_iterations(self):
        return len([x for x in os.listdir(os.path.join(self.project_dir, 'dlc-models'))])

    def get_image_dir_train(self):
        return f'./data/{self.db}/train/'

    def get_image_dir_val(self):
        return f'./data/{self.db}/val/'

    def get_vid_dir(self):
        return f'./data/{self.db}/videos'

    def get_boolean_keypoints(self):
        return [1 if x in self.keypoints else 0 for x in self.all_keypoints]

    def get_visibility(self, visibility):
        if visibility == 'Only_visible_keypoints':
            return True
        elif visibility == 'All_keypoints':
            return False
