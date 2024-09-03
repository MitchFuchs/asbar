# ASBAR: an Animal Skeleton-Based Action Recognition framework

## Paper

### Abstract
To date, the investigation and classification of animal behaviors have mostly relied on direct
human observations or video recordings with posthoc analysis, which can be labor-intensive,
time-consuming, and prone to human bias. Recent advances in machine learning for
computer vision tasks, such as pose estimation and action recognition, thus have the
potential to significantly improve and deepen our understanding of animal behavior.
However, despite the increased availability of open-source toolboxes and large-scale datasets
for animal pose estimation, their practical relevance for behavior recognition remains underexplored. In this paper, we propose an innovative framework, ASBAR, for Animal SkeletonBased Action Recognition, which fully integrates animal pose estimation and behavior
recognition. We demonstrate the use of this framework in a particularly challenging task: the
classification of great ape natural behaviors in the wild. First, we built a robust pose
estimator model leveraging OpenMonkeyChallenge, one of the largest available open-source
primate pose datasets, through a benchmark analysis on several CNN models from
DeepLabCut, integrated into our framework. Second, we extracted the great ape’s skeletal
motion from the PanAf dataset, a large collection of in-the-wild videos of gorillas and
chimpanzees annotated for natural behaviors, which we used to train and evaluate
PoseConv3D from MMaction2, a second deep learning model fully integrated into our
framework. We hereby classify behaviors into nine distinct categories and achieve a Top 1
accuracy of 74.98%, comparable to previous studies using video-based methods, while
reducing the model’s input size by a factor of around 20. Additionally, we provide an opensource terminal-based GUI that integrates our full pipeline and release a set of 5,440 keypoint
annotations to facilitate the replication of our results on other species and/or behaviors.

View online [here](https://elifesciences.org/reviewed-preprints/97962)

Download [here](https://github.com/elifesciences/enhanced-preprints-data/raw/master/data/97962/v1/97962-v1.pdf)

## Examples - walking, sitting, standing, running
<img src="https://github.com/MitchFuchs/asbar/assets/73831423/a95128ca-41c5-40c2-99bb-0eaaf04b0b5f.type" width="400" height="200">
<img src="https://github.com/MitchFuchs/asbar/assets/73831423/10316b38-0796-4c55-ab57-e0ff6bcfebb3.type" width="400" height="200">
<img src="https://github.com/MitchFuchs/asbar/assets/73831423/d5c46f70-4802-4dfb-b6d0-4f1898d3c990.type" width="400" height="200">
<img src="https://github.com/MitchFuchs/asbar/assets/73831423/b3472629-36f4-46d1-8202-e2f4f3690575.type" width="400" height="200">


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

Michael Fuchs, Emilie Genty, Klaus Zuberbühler, Paul Cotofrei (2024) ASBAR: an Animal Skeleton-Based Action Recognition framework. Recognizing great ape behaviors in the wild using pose estimation with domain adaptation. eLife13:RP97962
https://doi.org/10.7554/eLife.97962.1

```BibTeX
 @article{Fuchs_2024, title={ASBAR: an Animal Skeleton-Based Action Recognition framework. Recognizing great ape behaviors in the wild using pose estimation with domain adaptation}, url={http://dx.doi.org/10.7554/eLife.97962.1}, DOI={10.7554/elife.97962.1}, publisher={eLife Sciences Publications, Ltd}, author={Fuchs, Michael and Genty, Emilie and Zuberbühler, Klaus and Cotofrei, Paul}, year={2024}, month=aug }
```
