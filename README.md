# DDAD/Arogverse/DrivingStereo 데이터셋 학습
먼저 이미지 크기가 달라질 때의 사이즈(코드 상에서 kb crop size, img_scale 등) 수정이 필요합니다.
그 외에는 이미지와 라벨 각각의 폴더명, split을 정의하는 txt 파일, max_depth 값 등의 수정이 필요합니다.
저는 총 5곳에서 수정을 진행했습니다.

depth/datasets/ddad.py
depth/datasets/__init__.py
depth/datasets/pipelines/loading.py
configs/bts/bts_r50_ddad_24e.py
configs/_*base_*/datasets/ddad.py

그 외에도, 사실 Toolbox 자체에서 오류가 하나 있습니다.
depth/models/losses/sigloss.py
Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
Dg = torch.var(g, unbiased=False) + 0.15 * torch.pow(torch.mean(g), 2)
PyTorch에서는 var()의 디폴트 옵션이 unbiased=True인데요, 이 부분은 명시적으로 False로 지정하는 것이 수식적으로 올바릅니다. 이 부분은 KITTI에서는 딱히 차이가 없지만, DDAD 등의 다른 데이터셋에서는 학습의 안정성에서 큰 문제가 됨을 확인했습니다. 이건 제 생각인데요, random crop하면서 학습하니까 valid point 개수 N이 작아지는 케이스가 발생될 때 N으로 나눠주는 var() 함수에서 문제가 생기는 것 같습니다.

셋업이 끝난 후 학습은 이렇게 진행할 수 있습니다.
sh tools/dist_train.sh configs/depthformer/depthformer_swinl_22k_w7_ddad.py 4 --work-dir work_dirs/depthformer_swinl_22k_w7_ddad_n2





# Monocular-Depth-Estimation-Toolbox
## Introduction

Monocular-Depth-Estimation-Toolbox is an open source monocular depth estimation toolbox based on PyTorch and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v0.16.0.

It aims to benchmark MonoDepth methods and provides effective supports for evaluating and visualizing results.

<div align=center><img width="550" height="150" src="resources/teaser.gif"/></div>


### Major features
- **Unified benchmark**
  
  Provide a unified benchmark toolbox for various depth estimation methods.
- **Modular design**
  
  Depth estimation frameworks are decomposed into different components. One can easily construct a customized framework by combining different modules.
- **Support of multiple methods out of box**
  
  I would like to reproduce some of the most excellent depth estimation methods based on this toolbox.
- **High efficiency**
  
  It seems that there are few depth estimation benchmarks, so I start this project and hope it is helpful for research.

Thanks to MMSeg, we own these major features. :blush:

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones (partially release):
- [x] ResNet (CVPR'2016)
- [x] EfficientNet (ICML'2019)
- [x] Vision Transformer (ICLR'2021)
- [x] Swin Transformer (ICCV'2021)
- I recommend cross-package import in config, so that you can utilize other backbone in MMcls, MMseg, *etc*. Refer to [introduction](https://zhuanlan.zhihu.com/p/436865195). I will add more backbones in the future.

Supported methods:
- [x] [BTS (Arxiv'2019)](configs/bts)
- [x] [Adabins (CVPR'2021)](configs/adabins)
- [x] [SimIPU (AAAI'2022)](configs/simipu)
- [x] [DPT (ICCV'2021) (TBD)](configs/dpt)
- [x] [DepthFormer (Arxiv'2022)](configs/depthformer)
- [x] [BinsFormer (Arxiv'2022)](configs/binsformer) (Early release. It may contain bugs or potential performance drop)

Supported datasets:
- [x] [KITTI](docs/dataset_prepare.md#KITTI)
- [x] [NYU-Depth-V2](docs/dataset_prepare.md#NYU)
- [x] [SUN RGB-D (Eval)](docs/dataset_prepare.md#SUNRGBD)
- [x] [Cityscapes](docs/dataset_prepare.md#Cityscapes)


## Installation

Please refer to [get_started.md](docs/get_started.md#installation) for installation and [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

We provide [train.md](docs/train.md) and [inference.md](docs/inference.md) for the usage of this toolbox. 

<!-- In the future, there will be tutorials for [customizing dataset (TODO)](docs/tutorials/customize_datasets.md), [designing data pipeline (TODO)](docs/tutorials/data_pipeline.md), [customizing modules (TODO)](docs/tutorials/customize_models.md), and [customizing runtime (TODO)](docs/tutorials/customize_runtime.md). We also provide [training tricks (TODO)](docs/tutorials/training_tricks.md). -->

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This repo benefits from awesome works of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [Adabins](https://github.com/shariqfarooq123/AdaBins),
[BTS](https://github.com/cleinc/bts). Please also consider citing them.


## Cite
```bibtex
@misc{lidepthtoolbox2022,
  title={Monocular Depth Estimation Toolbox},
  author={Zhenyu Li},
  howpublished = {\url{https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox}},
  year={2022}
}
```

## Changelog
- **Jul. 23, 2022**: Release codes for BinsFormer. Potential bugs or performance drops can exist, waiting for completion.
- **Jun. 5, 2022**: Add support for custom dataset training. Add a warmup interface for sigloss to help convergence as discussed in Issue [#20](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/issues/20). Enhance the DPT support and fix bugs in provided pre-trained models as reported in Issue [#23](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/issues/23). 
- **Apr. 16, 2022**: Finish most of docs and provide all pre-trained parameters. Release codes about BTS, Adabins, DPT, SimIPU, and DepthFormer. Support KITTI, NYU-v2, SUN RGB-D(eval), and CityScapes.

## TODO
- I will complete release of BinsFormer (On going).
- I would like to include self-supervised depth estimation methods, such as MonoDepth2.
