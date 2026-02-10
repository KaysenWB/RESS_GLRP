## RESS_GLMP

This is the implementation code for the paper "Enhancing Risk Perception by Integrating Ship Interactions in Multi-Ship Encounters: A Graph-Based Learning Method".

Implementation and performance comparison of several common deep learning networks for ship motion prediction, including LSTM, GRU, Seq2Seq, TCN, Transformer, and our network GLMP. The input data are processed AIS data containing motion information and adjacency matrix, processed with reference to https://github.com/KaysenWB/AIS-Process.git.

Based on the output of the motion prediction, we present codes for DCPA and TCPA to calculate and measure the error, serving the subsequent risk perception.

## Code Description

The code for sequence prediction is located in the `Ress_GLRP/` folder, while the main entry point is `main_run.py`. The training, testing, and prediction pipeline is implemented in `processor.py`, with data loading handled by **`dataloader.py`. The core network architecture is defined in `GLRP.py`. Ablation studies for key components (VAE framework, graph network, and sequential structure) are provided in `Ablation_models.py`, and several baseline models (LSTM, GRU, Seq2Seq, TCN, Transformer) are included in `Compared_models.py`. Utility functions, mainly for metric calculation, are in `utils.py`, and a simple visualization script is in `visualization.py`. Outside this folder, the `D_TCPA.py` file is used for calculating ship-ship risk values based on the prediction results.


## Environment Setup

**System Requirements**

- Operating System: Linux (Ubuntu 18.04+ recommended)
- Python 3.8 or higher
- CUDA 11.3+ (for GPU acceleration, optional)

**Dependencies**

- torch==2.8.0
- numpy==2.0.1
- pandas==2.3.3
- math==1.3.0
- pytorch_tcn==1.2.3
- matplotlib == 3.7.2
- torch-geometric == 2.3.2

## Network Structure
<img src="https://github.com/KaysenWB/RESS_GLRP/blob/main/Figures/Figure03.jpg?raw=true" width="95%" height="95%">
The network structure of GLRP. Observable trajectories are first entered into the self-attention block, and features of the last timestep are entered into VGAE. Two layers of graph convolution are applied in the VGAE encoder, followed by variational inference to obtain latent variables. The latent space is sampled K times and input into the decoder, which generates the predicted sequence step by step by a GRU cell. Finally, Risk analysis is conducted based on the predictions.

## Result
<img src="https://github.com/KaysenWB/RESS_GLRP/blob/main/Figures/Figure08.jpeg?raw=true" width="95%" height="95%">
Qualitative comparison of the trajectory prediction results of GLRP and compared other deep learning networks in a 32-32 step multi-ship encounter prediction. It displays the greater prediction ability of GLRP.


## Citation
If you find this repository useful in your research, please consider citing the following papers:
```
@article{yang2025enhancing,
  title={Enhancing risk perception by integrating ship interactions in multi-ship encounters: A Graph-based Learning method},
  author={Yang, Kaisen and Yang, Dong and Lu, Yuxu},
  journal={Reliability Engineering \& System Safety},
  pages={111150},
  year={2025},
  publisher={Elsevier}
}
```

## Contact
If you have any questions, feel free to contact Haoyi Zhou through Email (kaisen.yang@connect.polyu.hk). Pull requests are highly welcomed.

