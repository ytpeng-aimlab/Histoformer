# Histoformer-Histogram-based Transformer for Efficient Underwater Image Enhancement (Histoformer)
Histoformer-Histogram-based Transformer for Efficient Underwater Image Enhancement

## Architecture of Histoformer
<img src="./Figure/Architecture_histoformer.png" width = "800" height = "400" div align=center />

## Enviornment Requirements
1. Create a virtual environment using `virtualenv`.
    ```
    virtualenv -p python3 venv
    ```
2. Install the package. (The version may be various depends on your devices.)
    ```
    source venv/bin/activate
    pip install -r requirements.txt
    ```
## Testing
*  Pretrained models : from *[[Histoformer](https://drive.google.com/file/d/1pDk4z7PuovlXIqj2nT-ENRzxSF6QQPSy/view?usp=drive_link)]* and place it in ./checkpoints <br>

    ```
    python test.py
    ```
*  Change the `"get_test_set"` in test.py. <br>

## Training
    ```
    python train.py
    ```
*  Change the `"get_training_set"` in train.py. <br>
