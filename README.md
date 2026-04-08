# NuWa
A Deep Learning Framework for High-Resolution Taiwan Weather Forecasting by Global Knowledge Integration and Adaptive Fine-Tuning. This work is developed as part of the National Science and Technology Council (NSTC) project, "Research on Key Technologies of Generative AI for Solving Localization Needs."

<img width="2326" height="1226" alt="image" src="https://github.com/user-attachments/assets/b5343db6-835b-4bea-8c59-31b3cb09c85e" />

### Environment Setting

We recommend using **Conda** to manage your environment. Follow these steps to set up the environment and resolve specific package compatibility issues.

**1. Create and Activate Environment**
```bash
conda update conda
conda create -n new_aurora_env python=3.10 -y
conda activate new_aurora_env
```

**2. Install Dependencies**
```bash
# Install PyTorch with CUDA 11.8 support
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1 --extra-index-url https://download.pytorch.org/whl/cu118

# Install core packages and remaining dependencies
pip install pyyaml
conda env update -f environment.yml
```

**3. Compatibility Fix for `basicsr`**
You must manually patch the `basicsr` library within your environment to fix a `torchvision` import error:
* **File Path:** `(your_environment_path)/new_aurora_env/lib/python3.10/site-packages/basicsr/data/degradations.py`
* **Find:** `from torchvision.transforms.functional_tensor import rgb_to_grayscale`
* **Replace with:** `from torchvision.transforms.functional import rgb_to_grayscale`

### Dataset Download

Please use the provided scripts to download ERA5 data. We focus on high-resolution presets for the Taiwan region.

**Notes:**
* Ensure your grid size is divisible by the patch size (default is **4**).

```bash
# Download static variables
python download_era5_data/constant_download_era5.py --region local

# Download main variables
python download_era5_data/download_era5.py --region local --start YYYY/MM/DD --end YYYY/MM/DD
```

---

## Inference

You can run the high-resolution inference pipeline using the following scripts:

* **Bash:**
```bash
bash public_bash_scripts/forecasting.sh
```

* **Direct CLI (recommended for custom runs):**
```bash
python inference.py \
  --dictionary_path /path/to/codebook.pth \
  --checkpoint_path /path/to/model_checkpoint.pth \
  --data_dir /path/to/dataset_root \
  --predict_time 2024010100
```

### CLI Arguments

- **`--dictionary_path`**: Path to the codebook/dictionary weight file (`.pth`).
  e.g. ./NuWa/Dictionary-4year.pth

- **`--checkpoint_path`**: Path to the model checkpoint (`.pth`) used for inference.
  e.g. ./NuWa/NuWa-1year.pth

- **`--data_dir`**: Root directory of the dataset used by `inference.py`.

- **`--predict_time`**: Forecast initialization time(s) in `YYYYMMDDHH`. This selects which time(s) to run and plot.
  - Single time (one run): `--predict_time 2024010100`
  - Time range (start and end): `--predict_time 2024010100 2024010300`
