# MS-2D-TAN

More details please read [MS-2D-TAN](https://github.com/microsoft/2D-TAN/tree/ms-2d-tan).

## Framework

[![alt text](https://github.com/microsoft/2D-TAN/raw/ms-2d-tan/pipeline.jpg)](https://github.com/microsoft/2D-TAN/blob/ms-2d-tan/pipeline.jpg)

## Main Results

### Main results on Charades-STA

| Feature | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 |
| ------- | --------- | --------- | --------- | --------- |
| VGG     | 45.65     | 27.20     | 85.91     | 57.61     |
| C3D     | 41.10     | 23.25     | 81.53     | 48.55     |
| I3D     | 56.64     | 36.21     | 89.14     | 61.13     |
| I3D*    | 60.08     | 37.39     | 89.06     | 59.17     |

(I3D* represents I3D features finetuned on Charades)

### Main results on ActivityNet Captions

| Feature | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 | Rank5@0.3 | Rank5@0.5 | Rank5@0.7 |
| ------- | --------- | --------- | --------- | --------- | --------- | --------- |
| C3D     | 61.04     | 46.16     | 29.21     | 87.31     | 78.80     | 60.85     |
| I3D     | 62.09     | 45.50     | 28.28     | 87.61     | 79.36     | 61.70     |

### Main results on TACoS

| Feature | Rank1@0.1 | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 | Rank5@0.1 | Rank5@0.3 | Rank5@0.5 | Rank5@0.7 |
| ------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| VGG     | 50.64     | 43.31     | 35.27     | 23.54     | 78.31     | 66.18     | 55.81     | 38.09     |
| C3D     | 49.24     | 41.74     | 34.29     | 21.54     | 78.33     | 67.01     | 56.76     | 36.84     |
| I3D     | 48.66     | 41.96     | 33.59     | 22.14     | 75.96     | 64.93     | 53.44     |           |

## Requirements

```sh
$ pip install -r requirements.txt
```

## Quick Start

### Download Datasets

Please download the data from [box](https://rochester.box.com/s/swu6rlqcdlebvwml8dyescmi7ra0owc5) or [baidu](https://pan.baidu.com/s/1pwo6lK71_ebit_hWykvgqQ?pwd=1mw4) and save it to the `data` folder.

### Training

Run the following commands for training:

#### Table 1

```sh
$ python moment_localization/run.py --cfg experiments/charades/MS-2D-TAN-G-VGG.yaml --verbose --tag base
$ python moment_localization/run.py --cfg experiments/charades/MS-2D-TAN-G-C3D.yaml --verbose --tag base
$ python moment_localization/run.py --cfg experiments/charades/MS-2D-TAN-G-I3D.yaml --verbose --tag base
$ python moment_localization/run.py --cfg experiments/charades/MS-2D-TAN-G-I3D-Finetuned.yaml --verbose --tag base
```

#### Table 2

```sh
$ python moment_localization/run.py --cfg experiments/activitynet/MS-2D-TAN-G-C3D.yaml --verbose --tag base
$ python moment_localization/run.py --cfg experiments/activitynet/MS-2D-TAN-G-I3D.yaml --verbose --tag base
```

#### Table 3

```sh
$ python moment_localization/run.py --cfg experiments/tacos/MS-2D-TAN-G-VGG.yaml --verbose --tag base
$ python moment_localization/run.py --cfg experiments/tacos/MS-2D-TAN-G-C3D.yaml --verbose --tag base
$ $ python moment_localization/run.py --cfg experiments/tacos/MS-2D-TAN-G-I3D.yaml --verbose --tag base
$ python moment_localization/run.py --cfg experiments/tacos/MS-2D-TAN-G-C3D-H512N512K5A8k9L2.yaml --verbose --tag base
$ python moment_localization/run.py --cfg experiments/tacos/MS-2D-TAN-G-I3D-H512N512K5A8k9L2.yaml --verbose --tag base
```

### Evaluation

Download all the trained model from [box](https://rochester.box.com/s/pvfgay9e90cdvke5qpktewzl99g3l8o9) or [baidu](https://pan.baidu.com/s/1TGOlQyeppMvVSjNy_LiF5w?pwd=28i2) and save them to the `release_checkpoints` folder.

Then, run the following commands to evaluate our trained models:

#### Table 1

```sh
$ python moment_localization/run.py --cfg experiments/charades/MS-2D-TAN-G-VGG.yaml --verbose --split test --mode test
$ python moment_localization/run.py --cfg experiments/charades/MS-2D-TAN-G-C3D.yaml --verbose --split test --mode test
$ python moment_localization/run.py --cfg experiments/charades/MS-2D-TAN-G-I3D.yaml --verbose --split test --mode test
$ python moment_localization/run.py --cfg experiments/charades/MS-2D-TAN-G-I3D-Finetuned.yaml --verbose --split test --mode test
```

#### Table 2

```
python moment_localization/run.py --cfg experiments/activitynet/MS-2D-TAN-G-C3D.yaml --verbose --split test --mode test
python moment_localization/run.py --cfg experiments/activitynet/MS-2D-TAN-G-I3D.yaml --verbose --split test --mode test
```

#### Table 3

```sh
$ python moment_localization/run.py --cfg experiments/tacos/MS-2D-TAN-G-VGG.yaml --verbose --split test --mode test
$ python moment_localization/run.py --cfg experiments/tacos/MS-2D-TAN-G-C3D.yaml --verbose --split test --mode test
$ python moment_localization/run.py --cfg experiments/tacos/MS-2D-TAN-G-I3D.yaml --verbose --split test --mode test
$ python moment_localization/run.py --cfg experiments/tacos/MS-2D-TAN-G-C3D-H512N512K5A8k9L2.yaml --verbose --split test --mode test
$ python moment_localization/run.py --cfg experiments/tacos/MS-2D-TAN-G-I3D-H512N512K5A8k9L2.yaml --verbose --split test --mode test
```

## The results we reproduced

### Reproduce in TACoS dataset


|   Model    | visual_embed | R1@0.1 | R1@0.3 | R1@0.5 | R1@0.7 | R5@0.1 | R5@0.3 | R5@0.5 | R5@0.7 |
| :--------: | :----------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|   paper    |     C3D      | 49.24  | 41.74  | 34.29  | 21.54  | 78.33  | 67.01  | 56.76  | 36.84  |
| reproduced |     C3D      | 46.49  | 39.04  | 31.34  | 22.39  | 75.78  | 64.46  | 52.61  | 36.27  |
|            |              |        |        |        |        |        |        |        |        |
|   paper    |     I3D      | 48.66  | 41.96  | 33.59  | 22.14  | 75.96  | 64.93  | 53.44  | 36.12  |
| reproduced |     I3D      | 47.99  | 41.59  | 33.49  | 22.87  | 76.16  | 65.98  | 55.49  | 37.72  |
|            |              |        |        |        |        |        |        |        |        |
|   paper    |     VGG      | 50.64  | 43.31  | 35.27  | 23.54  | 78.31  | 66.18  | 55.81  | 38.09  |
| reproduced |     VGG      | 50.69  | 42.01  | 33.04  | 21.94  | 80.10  | 69.01  | 55.56  | 37.29  |







