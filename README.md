<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# FlowDec

FlowDec ([ICLR 2025](https://openreview.net/forum?id=uxDFlPGRLX)) is a full-band audio codec for general audio sampled at 48 kHz that combines non-adversarial codec training with a stochastic postfilter based on a novel conditional flow matching method.

## Demo

See our demo page [here](https://sp-uhh.github.io/FlowDec/).

## News
- 2025/03/03 First version is released

## Installation

Create a new virtual environment (we recommend Python 3.10) and run
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```
(or whatever matches your local CUDA version).

## Checkpoints

You can find the checkpoints for FlowDec-75m and FlowDec-75s, as well as the weights for the underlying NDAC codecs NDAC-75 and NDAC-25, [here](https://github.com/facebookresearch/FlowDec/releases/download/checkpoints/checkpoints.zip).

## Inference

Please check out the notebook `demo.ipynb` for how to run inference using the pretrained checkpoints.

## Training

We use Hydra for model configuration and training. For training config files, see the `config/` folder.

### Data preparation

**NOTE:** We do not provide training/validation/test datasets here, so the training configurations in `config/` all use a dummy datamodule config `config/datamodule/example.yaml`. To actually train FlowDec, you should pre-enhance your own  dataset(s) with a pre-trained underlying codec, save the results as .wav files, and store the paired paths in a text file. You can for instance use our pre-trained NDAC variants - see the "Inference" section for how to run them.

The expected input format for FlowDec datasets is a file containing a comma-separated list of paths, e.g.:
```
/clean_path/file1.wav,/codec_output_path/file1.wav
/clean_path/file2.wav,/codec_output_path/file2.wav
[...]
```
where you would then have `train.txt`, `validation.txt` and `test.txt` each of this format, and adapt the datamodule config file to use these three .txt files instead of the dummy file.

### Running training

After modifying the datamodule, you can then for example run:

```bash
python train.py --config-name flowdec_75m
```

### Frequency-dependent sigma_y

For automatically determining the frequency-dependent sigma_y (see Section 3.5 in our [paper](https://openreview.net/pdf?id=uxDFlPGRLX)), you can use the helper script `scripts/estimate_flowdec_params.py`. This script also implements the heuristic for a global sigma_y discussed in our Appendix A.1.


## Citation

If you use our models, methods, or any derivatives thereof, please cite our [paper](https://openreview.net/forum?id=uxDFlPGRLX):

```
@inproceedings{
    welker2025flowdec,
    title={{FlowDec}: A flow-based full-band general audio codec with high perceptual quality},
    author={Simon Welker and Matthew Le and Ricky T. Q. Chen and Wei-Ning Hsu and Timo Gerkmann and Alexander Richard and Yi-Chiao Wu},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=uxDFlPGRLX}
}
```

## License
The majority of FlowDec is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [conditional-flow-matching](https://github.com/atong01/conditional-flow-matching), [sgmse](https://github.com/sp-uhh/sgmse), [BioinfoMachineLearning](https://github.com/BioinfoMachineLearning/bio-diffusion/tree/1cfc969193ee9f32d5300c63726b33a2a3b071d9), [audiotools](https://github.com/descriptinc/audiotools), and [descript-audio-code](https://github.com/descriptinc/descript-audio-codec) are licensed MIT; [NCSN++](https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py) is licensed Apache 2.0.
