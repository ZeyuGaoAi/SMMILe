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
1. Embedding Extraction (Customized)
```
python feature_extraction.py --set_your_params_here
```
You can also use the standard preprocessing pipeline provided by [CLAM](https://github.com/mahmoodlab/CLAM).

2. Superpixel Generation
```
python superpixel_generation.py --set_your_params_here
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
