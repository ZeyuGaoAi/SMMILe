# SMMILe
Accurate spatial quantification in computational pathology with multiple instance learning
![Graphic](Graphic.pdf)

# Installation

For HPC:
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

# Usage
## Preprocessing
1. Embedding Extraction
```
python feature_extraction.py --set_your_params_here
```
2. Superpixel Generation
```
python superpixel_generation.py --set_your_params_here
```

Binary or Multi-class dataset: ``` cd single/ ```
Multi-label dataset: ``` cd multi/ ```
## Training

1. Setup the config of stage 1, for example, ```./single/configs/config_renal_smmile_r1.yaml```
```
python main.py --config ./configs/config_renal_smmile_r1.yaml
```
2. Setup the config of stage 2, for example, ```./single/configs/config_renal_smmile_r1.yaml```
```
python main.py --config ./configs/config_renal_smmile_r2.yaml
```

## Evaluation
The whole test set:
```
python eval.py --data_root_dir /path/to/extracted/embedding/folder/ --results_dir /path/to/trained/model/folder/
```
The single WSI demo (several paths need to be set in demo.py):
```
python demo.py
```

# License

This project is licensed under the GPLv3 License and is available for non-commercial academic purposes.

# Acknowledgements

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
