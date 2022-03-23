# [CVPR 2022] Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation

[]()
Jiankun Li, Peisen Wang, Pengfei Xiong, Tao Cai, Ziwei Yan, Lei Yang, Jiangyu Liu, Haoqiang Fan, Shuaicheng Liu

**[arXiv](https://arxiv.org/abs/2203.11483) | [BibTeX](#citation)** 

<img src="img/teaser.jpg">

## Datasets

### The Proposed Dataset

#### Download

There are **two ways** to download the dataset(~400GB) proposed in our paper: 

- Download using shell scripts `dataset_download.sh`

```shell
sh dataset_download.sh
```

 - Download from BaiduCloud [here](https://pan.baidu.com/s/1iB96-ftCgPFTlrj220qw8Q)(Extraction code: aa3g) and extract the tar files manually.

#### Disparity Format

The disparity is saved as `.png` uint16 format which can be loaded using opencv `imread` function:

```python
def get_disp(disp_path):
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    return disp.astype(np.float32) / 32
```

### Other Public Datasets

Other public datasets we use including 

 - [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
 - [Sintel](http://sintel.is.tue.mpg.de/stereo)
 - [Middlebury](https://vision.middlebury.edu/stereo/data/)
 - [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-training-data)
 - [KITTI 2012/2015](http://www.cvlibs.net/datasets/kitti/eval_stereo.php) 
 - [Falling Things](https://research.nvidia.com/publication/2018-06_Falling-Things)
 - [InStereo2K](https://github.com/YuhuaXu/StereoDataset)
 - [HR-VS](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view)

## Citation
If you find this work helpful in your research, please cite:
```
@misc{Li2022crestereo,
      title={Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation},
      author={Jiankun Li and Peisen Wang, Pengfei Xiong and Tao Cai and Ziwei Yan and Lei Yang and Jiangyu Liu and Haoqiang Fan and Shuaicheng Liu},
      year={2022},
      eprint={2203.11483},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
