# perf_resnet8/ — ResNet-8 / CIFAR-10 veri konumu
perf_resnet8_pipe'in relative include ile kullandigi `resnet8_cifar_data.h` (resmi MLPerf-Tiny
pretrained agirliklar, BN-katlanmis + 8 test goruntusu) buraya gelir. Header buyuk (~1.1 MB)
oldugu icin repoya konmaz; uretimi:
  1) resmi modeli indir: github.com/mlcommons/tiny -> benchmark/training/image_classification/
     trained_models/pretrainedResnet.h5  (FP32 Keras, ~1.1 MB)
  2) ~/thesis/venv_thesis/bin/python example_model/gen_resnet8_cifar.py --h5 <pretrainedResnet.h5>
     (BN fold + preprocessing auto-detect [raw 0-255] + logit-seviyesi numpy self-check + header)
