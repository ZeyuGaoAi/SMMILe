SMMILe <img src="SmmileIcon.jpg" width="140px" align="right" />
===========
## Accurate spatial quantification in computational pathology with multiple instance learning.
![Graphic](SMMILe-graphical-abstract.png)
## Framework
![Graphic](SMMILeGraphic.png)

# Installation

For HPC NVIDIA A100:
```
module purge
module load GCCcore/11.3.0 Python/3.10.4
module load GCCcore/11.3.0 OpenSlide/3.4.1-largefiles
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
python -m venv ./pyvenv/smmile
source ./pyvenv/smmile/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Demo Datasets and Models

The original WSI files (.svs) of TCGA data can be downloaded from [GDC Portal page](https://portal.gdc.cancer.gov/v1/repository).

We provide extracted embeddings (ResNet-50, Conch) and superpixel segmentation results on our [Hugging Face dataset page](https://huggingface.co/datasets/zeyugao/SMMILe_Datasets).


# Usage
## Preprocessing
### 1. Embedding Extraction (Customized)
```
python feature_extraction.py --encoder_name {resnet50/conch} \
        --feature_dir /path/to/your/embedding/save/folder\
        --anno_dir /path/to/annotation/folder/\
        --wsi_dir /path/to/svs/file/folder/\
        --file_list_path /path/to/slide/list/file/\
        --patch_size 512 --step_size 512 --level 1\
```
For TCGA datasets (Renal, Lung, Gastric), level = 1.

For Camelyon-16 dataset (Breast), UBC-OCEAN (Ovarian), level = 0.

For datasets with tessellated patches (SICAPv2), patches are organized into subdirectories based on their class labels. Each parent folder represents a WSI, and its subfolders contain patches belonging to different classes.:
```
/xxx/SICAPv2/
│── 17B00208864/                # WSI ID (Parent folder for a slide)
│   ├── 0/                      # Class 0 patches
│   │   ├── patch_001.png
│   │   ├── patch_002.png
│   │   ├── ...
│   ├── 3/                      # Class 3 patches
│   │   ├── patch_101.png
│   │   ├── patch_102.png
│   │   ├── ...
│   ├── 4/                      # Class 4 patches
│   │   ├── patch_201.png
│   │   ├── patch_202.png
│   │   ├── ...
```
```
python feature_extraction_patch.py --encoder_name {resnet50/conch} \
        --feature_dir /path/to/your/embedding/save/folder\
	--patch_dirs /path/to/patches/*
	--file_suffix 0_1024.npy --patch_size 1024
```


You can also use the standard preprocessing pipeline provided by [CLAM](https://github.com/mahmoodlab/CLAM). 
We have a modified version [CLAM_Pre](https://github.com/ZeyuGaoAi/CLAM_PreProcessing) to generate embedding files with readable formats for SMMILe.

### 2. Superpixel Generation
Set up the size (patch size) as the same as the feature extraction step, n_segments_persp can be set to 9, 16, and 25 for different datasets. We use 16 as default.
```
python superpixel_generation.py --size 2048 --n_segments_persp 16 --compactness 50 \
                                --file_suffix '*0_2048.npy' --keyword_feature feature \
                                --fea_dir /path/to/your/embedding/save/folder \
                                --sp_dir /path/to/your/superpixel/save/folder/sp_n%d_c%d_%d/
```

## Training

Binary or Multi-class dataset: ``` cd single/ ```
Multi-label dataset: ``` cd multi/ ```

1. Setup the config of stage 1, for example, ```./single/configs_rcc/config_renal_smmile_r1_conch.yaml```, the current config is set for the base version without any module. 
```
python main.py --config ./configs_rcc/config_renal_smmile_r1_conch.yaml \
               --drop_with_score --D 1 --superpixel --exp_code smmile_d1sp \
               --max_epochs 40
```
2. After stage 1, setup the config of stage 2, for example, ```./single/configs_rcc/config_renal_smmile_r1_conch.yaml```
```
python main.py --config ./configs_rcc/config_renal_smmile_r1_conch.yaml  --drop_with_score --D 1 \
               --superpixel --inst_refinement --mrf --exp_code smmile_d1sp_ref_mrf \
               --models_dir /home/z/zeyugao/SMMILe/single/results_conch_rcc/smmile_d1sp_s1 \
               --max_epochs 20
```
Note that using ```--consistency``` for the dataset containing normal cases in both stages.
Also, ```--mrf``` is not suitable for datasets only with small tumor regions, like Camelyon16, most WSIs only have several patches containing tumor.

## Evaluation
The whole test set:
```
python eval.py --data_root_dir /path/to/extracted/embedding/folder/ \
               --data_sp_dir /path/to/superpixels/folder/ \
               --results_dir /path/to/trained/models/folder/ \
               --models_exp_code smmile_d1sp_ref_mrf_s1 --save_exp_code _conch_rcc
```
Heat map generation:
```
python generate_heatmap.py \
    --model_name smmile \
    --wsi_dir '/path/to/original/svs_file/folder/*.svs' \
    --results_dir '/path/to/generated/results/folder/' \
    --num_workers 8
```
The single WSI demo (several paths need to be set in demo.py):
```
python demo.py
```

# License

This project is licensed under the GPLv3 License and is available for non-commercial academic purposes.

# Acknowledgements

We acknowledge funding and support from Cancer Research UK and the Cancer Research UK Cambridge Centre [CTRQQR-2021-100012], The Mark Foundation for Cancer Research [RG95043], GE HealthCare, and the CRUK National Cancer Imaging Translational Accelerator (NCITA) [A27066]. Additional support was also provided by the National Institute of Health Research (NIHR) Cambridge Biomedical Research Centre [NIHR203312] and EPSRC Tier-2 capital grant [EP/P020259/1]. Calculations were performed in part using the Sulis Tier 2 HPC platform hosted by the Scientific Computing Research Technology Platform at the University of Warwick. Sulis is funded by EPSRC Grant EP/T022108/1 and the HPC Midlands+ consortium. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

# Citation

If you find any useful, please cite our paper.

Gao Z, Mao A, Dong Y, et al. Accurate spatial quantification in computational pathology with multiple instance learning[J]. medRxiv, 2024: 2024.04. 25.24306364.s

```
@article {Gao2024,
  title={Accurate spatial quantification in computational pathology with multiple instance learning},
  author={Gao, Zeyu and Mao, Anyu and Dong, Yuxing and Wu, Jialun and Liu, Jiashuai and Wang, Chunbao and He, Kai and Gong, Tieliang and Li, Chen and Crispin-Ortuzar, Mireia},
	year = {2024},
	doi = {10.1101/2024.04.25.24306364},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2024/04/26/2024.04.25.24306364},
	journal = {medRxiv}
}
```
