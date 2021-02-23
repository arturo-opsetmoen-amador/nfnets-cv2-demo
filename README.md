# Code to run a demo of nfnets with cv2

This repository contains code to run the newly released NFNet model. The code is heavily based in Google's 
implementation for the ICLR 2021 paper
["Characterizing signal propagation to close the performance gap in unnormalized
ResNets,"](https://arxiv.org/abs/2102.06171) by Andrew Brock, Soham De, and
Samuel L. Smith, and the arXiv preprint ["High-Performance Large-Scale Image
Recognition Without Normalization"](http://dpmd.ai/06171) by
Andrew Brock, Soham De, Samuel L. Smith, and Karen Simonyan.

For more details, visit [deep-mind's github repository](https://github.com/deepmind/deepmind-research/tree/master/nfnets)


@inproceedings{brock2021characterizing,
  author={Andrew Brock and Soham De and Samuel L. Smith},
  title={Characterizing signal propagation to close the performance gap in
  unnormalized ResNets},
  booktitle={9th International Conference on Learning Representations, {ICLR}},
  year={2021}
}

Adaptive Gradient Clipping (AGC) and the NFNets models:

@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:2102.06171},
  year={2021}
}

## Running this demo

### Setup
Use the setup.sh scrip to get the requirements with pip. This script will also download 
the F0 NTNet implementation from [this link.](https://storage.googleapis.com/dm-nfnets/F0_haiku.npz)

### Running NFNet on your webcam feed with CV2
To run the demo, check the output of `echo $DISPLAY` in your console and modify line 
13 in `main.py` (`os.environ['DISPLAY'] = `). Also, be sure that line 16 in `main.py` 
points to your webcam capture device (`cap = cv2.VideoCapture(0)`).