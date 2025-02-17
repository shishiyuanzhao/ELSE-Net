# ELSE-Net
This is the official implementation of our conference paper : "[Excluding the impossible for Open Vocabulary Semantic Segmentation]"
## Introduction
This paper a new open vocabulary semantic segmentation method (Excluding the ImpossibLe Semantic SEgmentation Network, ELSE-Net) is proposed. This method mainly uses the General Processing Block (GP-Block) to generate a class-agnostic mask, and then calculates the inclusion probability (the probability of belonging to a certain category); and uses the Excluding the Impossible Block (EXP-Block) to generate a high-quality exclusion probability (the probability of not belonging to a certain category), uses the exclusion probability to correct the inclusion probability, and uses the corrected probability to accurately annotate the class-agnostic mask. In addition, the EXP-Block in this method is model-agnostic and can be integrated into the existing forward method to improve performance.
### Installation
1. Clone the repository
    ```sh
    git clone https://github.com/shishiyuanzhao/ELSE-Net.git
    ```
2. Navigate to the project directory
    ```sh
    cd ELSE-Net
    ```
3. Install the dependencies
    ```sh
    bash install.sh
    ```
### Data Preparation
The data should be organized like:
datasets/
    coco/
        ...
        train2017/
        val2017/
        stuffthingmaps_detectron2/
    VOC2012/
        ...
        images_detectron2/
        annotations_detectron2/
    pcontext/
        ...
        val/
    pcontext_full/
        ...
        val/
    ADEChallengeData2016/
        ...
        images/
        annotations_detectron2/
    ADE20K_2021_17_01/
        ...
        images/
        annotations_detectron2/        
```
### Note
In the code, those datasets are registered with their related dataset names. The relationship is:
```
coco_2017_*_stuff_sem_seg : COCO Stuff-171
voc_sem_seg_*: Pascal VOC-20
pcontext_sem_seg_*: Pascal Context-59
ade20k_sem_seg_*: ADE-150
pcontext_full_sem_seg_*： Pascal Context-459
ade20k_full_sem_seg_*: ADE-847
### Usage
- #### Evaluation
 - evaluate trained model on validation sets of all datasets.
  python train_net.py --eval-only --config-file <CONFIG_FILE> --num-gpus <NUM_GPU> OUTPUT_DIR <OUTPUT_PATH> MODEL.WEIGHTS <TRAINED_MODEL_PATH>
  - evaluate trained model on validation sets of one dataset.
  python train_net.py --eval-only --config-file <CONFIG_FILE> --num-gpus <NUM_GPU> OUTPUT_DIR <OUTPUT_PATH> MODEL.WEIGHTS <TRAINED_MODEL_PATH> DATASETS.TEST "('<FILL_DATASET_NAME_HERE>',)"
- #### Training
-     python train_net.py --config-file <CONFIG_FILE> --num-gpus <NUM_GPU> OUTPUT_DIR <OUTPUT_PATH> WANDB.NAME <WANDB_LOG_NAME>
### Cite 
If you find it helpful, you can cite our paper in your work.
```
@proceedings{yuan2025Else,
  title={Excluding the impossible for Open Vocabulary Semantic Segmentation},
  author={Shiyuan Zhao, Baodi Liu, Yu Bai, Weifeng Liu, Shuai Shao},
  journal={AAAI},
  year={2025}
}
```
