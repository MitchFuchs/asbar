import os
import pickle
import numpy as np
from tools.dlc_utils import DLC


class MMA(DLC):
    def __init__(self, args):
        super().__init__(args)
        self.db = args.dataset
        self.mm_dataset = args.mm_dataset
        self.data_dir = self.get_data_dir()
        if self.db is not None:
            self.activities = self.get_activities()
            # self.activities = None
            self.labels = self.get_labels()
            self.split_dir = self.get_split_dir()
            self.dataset_split = self.get_dataset_split()
            self.all_videos = self.get_all_videos()
            self.all_pickles = self.get_all_pickles()
        self.n_keypoints = len(self.cfg['multianimalbodyparts'])
        self.mm_cfg = {
            'sample_interval': 20,
            'sequence_length': 20,
            'activity_duration_threshold': 20
        } #add in conf.yaml?

    def create_training_dataset(self):
        self.all_pickles = self.get_all_pickles()
        self.activities = self.get_activities()

        for label_id, label in enumerate(self.labels):
            for activity in self.activities[label]:
                name = activity[0]
                if name in self.dataset_split['test']:
                    split = 'test'
                elif name in self.dataset_split['val']:
                    split = 'val'
                else:
                    split = 'train'
                self.create_skeleton_data(split, label_id, label, activity)
                
        splits = ['train', 'val', 'test']

        for split in splits:
            data = []
            files = [x for x in os.listdir(os.path.join(self.data_dir, split)) if x.endswith('pkl')]
            for file in files:
                with open(os.path.join(self.data_dir, split, file), 'rb') as fp:
                    activity = pickle.load(fp)
                    data.append(activity[0])

            filename = os.path.join(self.data_dir, f"{split}.pkl")

            with open(filename, 'wb') as f:
                pickle.dump(data, f)

    def create_skeleton_data(self, split, label_id, label, activity):
        name, ape_id, start, stop, bboxes = activity
        sample_interval, sequence_length, activity_duration_threshold = self.mm_cfg.values()
        full_pickle = self.get_full_pickle(name)
        full_pk = os.path.join(self.dest_folder, full_pickle)
        with open(full_pk, 'rb') as fp:
            kp = pickle.load(fp)

        total_frames = stop - start + 1

        if total_frames >= activity_duration_threshold:
            coords = np.zeros((1, total_frames, self.n_keypoints, 2))  # shape 1xTxKx2
            scores = np.zeros((1, total_frames, self.n_keypoints))  # shape 1xTxK
            num_frames_in_dlc = max(len(kp.keys()) - 1, 1)
            add_n_zero = int(np.ceil(np.log10(num_frames_in_dlc)))
            keys = ['frame' + str(x).zfill(add_n_zero) for x in range(start - 1, stop)]

            for i, key in enumerate(keys):
                if key in kp.keys():
                    coordinates = kp[key]['coordinates'][0]
                    confidence = kp[key]['confidence']
                    # costs = kp[key]['costs']

                    xmin, ymin, xmax, ymax = bboxes[i]

                    for j, bdp in enumerate(coordinates):
                        if len(bdp) >= 1:
                            k_in = []
                            for k, xy in enumerate(bdp):
                                # if coordinates are in bbox
                                if xmin < xy[0] < xmax and ymin < xy[1] < ymax:
                                    k_in.append(k)  # add index of a pair of coordinates to temporary list
                            # if more than one pair of coordinates for that bodypart in bbox, keep only the one with
                            # max confidence
                            if len(k_in) > 1:
                                confidence_in = [confidence[j][x][0] for x in
                                                 k_in]  # confidences of coordinates in bbox
                                conf_max = np.argmax(confidence_in)  # index of max confidence of coordinates in bbox
                                k_in = [k_in[conf_max]]  # final index of coordinates

                            if len(k_in) == 1:
                                coords[0][i][j][:] = bdp[k_in[0]]
                                scores[0][i][j] = confidence[j][k_in[0]]
                else:
                    print('error, verify the corresponding sample:', label, name, key)

            sequence = 0
            ext, shape = self.get_meta(name, full_pk)

            for i in range(0, total_frames, sample_interval):
                if total_frames - i >= sequence_length:
                    filename = '_'.join(
                        [name, str(label_id), str(sequence), str(i), str(i + sequence_length)])

                    out = {}
                    out['keypoint'] = coords[:, i:i + sequence_length, :, :]
                    out['keypoint_score'] = scores[:, i:i + sequence_length, :]
                    out['frame_dir'] = filename + ext
                    out['label'] = label_id
                    out['img_shape'] = shape
                    out['original_shape'] = shape
                    out['total_frames'] = sequence_length

                    filename = os.path.join(self.data_dir, split, filename + '.pkl')

                    sequence += 1
                    with open(filename, 'wb') as f:
                        pickle.dump([out], f)
                # else:
                #     print('did not pass', stop - i + 1)

    def get_meta(self, name, full_pk):
        ext = [os.path.join(self.vid_dir, x) for x in self.all_videos if x[:len(name)] == name][0][-4:]

        meta_pk = full_pk.replace('_full.pickle', '_meta.pickle')

        with open(meta_pk, 'rb') as fp:
            meta = pickle.load(fp)
            shape = meta['data']['frame_dimensions']

        return ext, shape

    def get_all_videos(self):
        return [x for x in os.listdir(self.vid_dir)]

    def get_all_pickles(self):
        return [x for x in os.listdir(self.dest_folder) if x.endswith('full.pickle')]

    def get_full_pickle(self, name):
        snap = self.snapshot.split('-')[1]
        pickles_from_video = [x for x in self.all_pickles if x[:len(name)+3] == name+'DLC']
        full_pickle = [x for x in pickles_from_video if x.endswith(snap+'_full.pickle')]
        assert len(full_pickle) == 1, f'Multiple pickle files {len(full_pickle)} found for {name} - {full_pickle}.'
        return full_pickle[0]

    def get_data_dir(self):
        data_dir = os.path.join(os.getcwd(), 'models', 'mmaction2', 'data', 'posec3d',
                                self.project + '-' + self.iteration)
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
            os.mkdir(os.path.join(data_dir, 'train'))
            os.mkdir(os.path.join(data_dir, 'val'))
            os.mkdir(os.path.join(data_dir, 'test'))
        return data_dir

    def get_activities(self):
        file = os.path.join(f'./data/{self.db}/activities')
        assert os.path.isfile(file), 'activities not found'
        with open(file, 'rb') as fp:
            activities = pickle.load(fp)
        return activities

    def get_labels(self):
        labels = list(self.activities.keys())
        with open(os.path.join(self.data_dir, f'label_map_{self.db}.txt'), 'w') as f:
            for label in labels:
                f.write(f"{label}\n")
        return labels

    def get_split_dir(self):
        return f'./data/{self.db}/splits/'

    def get_dataset_split(self):
        data = {}
        with open(os.path.join(self.split_dir, 'trainingdata.txt')) as f:
            data['train'] = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.split_dir, 'validationdata.txt')) as f:
            data['val'] = [line.strip() for line in f.readlines()]
        with open(os.path.join(self.split_dir, 'testdata.txt')) as f:
            data['test'] = [line.strip() for line in f.readlines()]

        return data
