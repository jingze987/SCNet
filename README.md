# <p align=center>`Learning Structural Consensus Network for Cross-Domain Co-Salient Object Detection`</p>
> **Authors:**
> Jingze Liu<sup>1</sup>,
> Guanting Guo<sup>1</sup>,
> Huihui Song<sup>1</sup>,
> Kaihua Zhang<sup>2</sup>,
> Guangcan Liu<sup>2</sup>, 
> Pengfei Zhu<sup>2</sup>.<br>
> <sup>1</sup>Nanjing University of Information Science and Technology<br>
> <sup>2</sup>School of Automation, Southeast University
## Abstract
Co-salient object detection (CoSOD) focuses on segmenting common and salient objects that share semantic attributes across multiple images. Despite considerable progress, prevailing methods are typically trained and evaluated within a single domain, which significantly limits their generalization in real-world scenarios characterized by diverse domains.
To tackle this challenge, we propose a Structural Consensus Network (SCNet) that considers  the domain-invariant shared geometric layout of the co-salient objects as prior to enhance the cross-domain generalization for CoSOD.
The SCNet incorporates two key components: a Structural Consensus Learning (SCL) module and a Structural Consensus Denoising (SCD) module. The SCL module adaptively learns structural consensus that encodes the geometric layout by emphasizing object boundaries of co-salient regions, while the SCD module leverages dynamic convolutional kernels as adaptive low-pass filters for local context learning that can effectively enhance the consensus compactness while preserving sharp object boundaries.
By cascading SCL and SCD, our SCNet effectively extracts high-quality structural consensus representations, achieving superior performance compared to a variety of state-of-the-art methods on cross-domain benchmark datasets.
<p align="center">
    <img src="figs/pipeline.jpg"/> <br />
    <em> 
    The pipeline of our SCNet.
    </em>
</p>

## Usage

**Environment**

```
Install Python 3.12  PyTorch 2.7.1
pip install -r requirements.txt
```

**Datasets preparation**
   The complete test datasets, including CoCA, CoSOD3k, and CoSal2015, can be [downloaded](https://pan.baidu.com/s/1PRpINBodW0yP4QTobceDEQ?pwd=aa4x) (aa4x) and placed in the `./test_data/test` directory.

   The directory structure of the dataset is as follows:

```
+-- SCNet
|   ...
|   +-- test_data
|       +-- test
|           +-- CoCA
|               +-- Image
|               +-- Corruption
|               +-- GroundTruth
|           +-- CoSal2015
|               +-- Image
|               +-- Corruption
|               +-- GroundTruth
|           +-- CoSOD3k
|               +-- Image
|               +-- Corruption
|               +-- GroundTruth
|   ...
```
After preparing the three datasets, run the following command to generate corrupted images.
```
python create_corruption_datasets.py
```
**Model Weights**

Download our trained model from [DUTS+CoCoSeg](https://pan.baidu.com/s/1cCJABB8htXAaO9gauKBSHg?pwd=s72j) (s72j) and put it in the project root directory.

**Test + Evaluate**

Run the following command to perform inference.
```
python test_demo.py --data_root './test_data/test' --pred_root './pred_dir' --ckpt_root './' --ckpt_name 'Best_DS.pth'
```