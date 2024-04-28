# SMMILe
Accurate spatial quantification in computational pathology with multiple instance learning

# Installation

For HPC:
```
module purge
module load GCCcore/8.3.0 Python/3.7.4
module load GCC/8.3.0 CUDA/10.1.243 cuDNN/7.6.4.38
python -m venv ./pyvenv/smmile
source ./pyvenv/smmile/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
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
