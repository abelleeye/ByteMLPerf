[ [中文](README.zh_CN.md) ]

# Tested Models

| Model name |  Precision | QPS | Dataset | Metric name | Metric value | report |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| bert-torch-fp32 | FP16 | 894 | OPEN_SQUAD | F1 Score | 85.81071 | [report](../../reports/DL/bert-torch-fp32/) |
| resnet50-torch-fp32 | FP16 | 12,121 | OPEN_IMAGENET | Top-1 | 0.7698 | [report](../../reports/DL/resnet50-torch-fp32/) |
| resnet50-torch-fp32 | INT8 | 23,021 | OPEN_IMAGENET | Top-1 | 0.7682 | [report](../../reports/DL/resnet50-torch-fp32/) |
| widedeep-tf-fp32 | FP32 | 11,085,440 | OPEN_CRITEO_KAGGLE | Top-1 | 0.77395 | [report](../../reports/DL/widedeep-tf-fp32/) |
| widedeep-tf-fp32 | FP16 | 14,974,208 | OPEN_CRITEO_KAGGLE | Top-1 | 0.77394 | [report](../../reports/DL/widedeep-tf-fp32/) |
| yolov5-onnx-fp32 | INT8 | 1,292 | FAKE_DATASET | | | [report](../../reports/DL/yolov5-onnx-fp32/) |
| open_waveglow-tf-fp32 | FP16 | 86,750 | FAKE_DATASET | | | [report](../../reports/DL/open_waveglow-tf-fp32/) |

# How to run

## Tested in the following environments：
Ubuntu20.04 & Python3.8.10


## Install python3.8.10 dependencies

```
python3 -m pip install -r byte_mlperf/requirements.txt
python3 -m pip install -r byte_mlperf/backends/DL/requirements.txt
```

## Environmental preparation

```
export PYTHONPATH=$PWD/byte_mlperf:$PYTHONPATH
```

## Run

```
python3 launch.py --task bert-torch-fp32  --hardware_type DL
```

## How to get SDK etc.

Please send to <a href="mailto:348321273@qq.com">348321273@qq.com</a>
