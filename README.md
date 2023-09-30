# ASBAR: an Animal Skeleton-Based Action Recognition framework

## Installation

Clone source code:

```
git clone https://github.com/MitchFuchs/asbar.git
```

Move into directory:

```
cd asbar
```

Create conda environment:
```
conda env create -f requirements.yaml
```

Activate conda environment:
```
conda activate asbar
```

Launch GUI:
```
./gui.sh
```
If the installation was successful, you should see this: 
![0_start](https://github.com/MitchFuchs/asbar/assets/73831423/e790819c-12fb-467b-8289-7cad180df279)


## Prepare Datasets
Follow the same structure as below:
### Pose dataset
    ├── ...
    ├── data                              # data folder
    │   ├── <pose_dataset>                # create a new folder and name it with the name of the pose dataset 
    |   │   ├── train                     # create a new folder named 'train' containing training images
    |   │   |   ├── train_0000.jpg        # training image (name irrelevant) 
    |   │   |   ├── train_0001.jpg        # training image (name irrelevant) 
    |   │   |   ├── train_0002.jpg        # training image (name irrelevant)
    │   |   |   └── ...                   # etc.
    |   │   ├── val                       # create a new folder named 'val' containing validation images
    |   │   |   ├── val_0000.jpg          # validation image (name irrelevant) 
    |   │   |   ├── val_0001.jpg          # validation image (name irrelevant) 
    |   │   |   ├── val_0002.jpg          # validation image (name irrelevant)
    │   |   |   └── ...                   # etc.
    |   │   ├── train_annotations.json    # add a json file with training annotations structured as below
    |   │   └── val_annotations.json      # add a json file with validation annotations structured as below
    │   └── ...                   
    └── ...

#### structure of train_annotations.json and val_annotations.json

```
{
    "data": [
        {
            "file": "train_0000.jpg",
            "species": "Rhesus_macaque",
            "bbox": [x, y, w, h],
            "landmarks": [x_1, y_1, x_2, y_2, ..., x_n, y_n],
            "visibility": [1, 1, ..., 1]
        },
        {
            "file": "train_0001.jpg",
            "species": "Gorilla",
            "bbox": [x, y, w, h],
            "landmarks": [x_1, y_1, x_2, y_2, ..., x_n, y_n],
            "visibility": [1, 0, ..., 0]
        }
    ]
}

```

### Behavior dataset
    ├── ...
    ├── data                              # data folder
    │   ├── <behavior_dataset>            # create a new folder and name it with the name of the behavior dataset 
    |   │   ├── splits                    # create a new folder named 'splits' containing data split files
    |   │   |   ├── trainingdata.txt      # text file listing all video names (without file extension) of the training dataset 
    |   │   |   ├── validationdata.txt    # text file listing all video names (without file extension) of the validation dataset 
    |   │   |   └── testdata.txt          # text file listing all video names (without file extension) of the test dataset 
    |   │   ├── videos                    # create a new folder named 'videos' containing all videos
    |   │   |   ├── 0000.mp4              # video file (name irrelevant) 
    |   │   |   ├── 0001.mp4              # video file (name irrelevant) 
    |   │   |   ├── 0002.mp4              # video file (name irrelevant)
    │   |   |   └── ...                   # etc.
    |   │   └── activities                # add a pickle file with behavior annotations structured as below
    │   └── ...                   
    └── ...

#### structure of activities

```
{
    "sitting": [                                                     # behavior
        [
            0000,                                                    # video name (without file extension) 
            0,                                                       # animal_id (can be left as 0 or for tracking id)
            126,                                                     # number of the first frame displaying the behavior
            166,                                                     # number of the last frame displaying the behavior 
            [(x_1, y_1, x_2, y_2), ..., (x_1, y_1, x_2, y_2)]        # list of bounding box coordinates for each frame, (x_top_left, y_top_left, x_bottom_right, y_bottom_right) 
        ],
        [
            0000,
            0,
            180,
            190,
            [(x_1, y_1, x_2, y_2), ..., (x_1, y_1, x_2, y_2)]
        ],
    ],
    "standing": [                  
        [
            0000,
            0,
            167,
            179,
            [(x_1, y_1, x_2, y_2), ..., (x_1, y_1, x_2, y_2)]
        ],
        [
            0001,
            0,
            23,
            58,
            [(x_1, y_1, x_2, y_2), ..., (x_1, y_1, x_2, y_2)]
        ]
    ]
}

```


## Reference

If you use this material, please cite it as below.

Michael Fuchs, Emilie Genty, Klaus Zuberbühler, Paul Cotofrei. “ASBAR: an Animal Skeleton-Based Action Recognition framework. Recognizing great ape behaviors in the wild using pose estimation with domain adaptation”. In: bioRxiv (2023). doi: 10.1101/2023.09.24.559236. eprint: https://www.biorxiv.org/content/early/2023/09/25/2023.09.24.559236.full.pdf. 

```BibTeX
@article {asbar_Fuchs2023,
	author = {Michael Fuchs and Emilie Genty and Klaus Zuberb{\"u}hler and Paul Cotofrei},
	title = {ASBAR: an Animal Skeleton-Based Action Recognition framework. Recognizing great ape behaviors in the wild using pose estimation with domain adaptation},
	elocation-id = {2023.09.24.559236},
	year = {2023},
	doi = {10.1101/2023.09.24.559236},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/09/25/2023.09.24.559236},
	eprint = {https://www.biorxiv.org/content/early/2023/09/25/2023.09.24.559236.full.pdf},
	journal = {bioRxiv}
}
```
