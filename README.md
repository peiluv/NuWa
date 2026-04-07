# NuWa
A Deep Learning Framework for High-Resolution Taiwan Weather Forecasting by Global Knowledge Integration and Adaptive Fine-Tuning. This work is developed as part of the National Science and Technology Council (NSTC) project, "Research on Key Technologies of Generative AI for Solving Localization Needs."

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
python download_era5_data/constant_download_era5.py --region tw

# Download main variables
python download_era5_data/download_era5.py --region tw --start YYYY/MM/DD --end YYYY/MM/DD
```

---

## Inference

You can run the high-resolution inference pipeline using the following scripts:

* **Bash:**
```bash
bash public_bash_scripts/NuWa_inference_custom_rollout_1hr.sh
```