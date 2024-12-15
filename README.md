# Video-Game-SR
Code for [Super Resolution in Video Games](https://www.kaggle.com/competitions/super-resolution-in-video-games)

## Guide to run the code

### Creating Patches

Run `create_patch.py` after changing the following variables

```python
hrImagePath = "/data4/super-resolution-in-video-games/train/hr"
lrImagePath = "/data4/super-resolution-in-video-games/train/lr"
allOriginalPatchSaveLoc = "/data3/ALLPatches_Vid_Game.h5"
hrPatchSize = 256
lrPatchSize = 64
```

### Visualizing Patches

Run `visualize.py` after changing the following variables

```python
file_path = '/data3/ALLPatches_DVI2K.h5'
```

### Networks

Network is defined in `testNet_5.py`

### Training

Run `train.py` after changing the following variables

```python
cuda_no = 0
dataset_file_path = '/data3/ALLPatches_Vid_Game.h5'
num_workers = 4
batch_size = 32
num_epoch = 30
```


### Testing

Run `test.py` after changing the following variables

```python
cuda_no = 2
trained_model_loc = "./TestNet5_Models/TestNet5_VidGame_ep_30.pt"
test_dataset_folder = "/data4/super-resolution-in-video-games/test/lr"
network_result_dir = "Network_Result"
```

### Generating Submission File

Run `enc.py` after changing the following variables

```python
finalResult = 'submission.csv'
imgDir = "./Network_Result"
```

### Supporting Files

```python
- dataset.py
- utils.py
```
