# SS-GLMP

This is the implementation code for the paper "Enhancing Risk Perception by Integrating Ship Interactions in Multi-Ship Encounters: A Graph-Based Learning Method".

Implementation and performance comparison of several common deep learning networks for ship motion prediction, including LSTM, GRU, Seq2Seq, TCN, Transformer, and our network GLMP. The input data are processed AIS data containing motion information and adjacency matrix, processed with reference to https://github.com/KaysenWB/AIS-Process.git.

Based on the output of the motion prediction, we present codes for DCPA and TCPA to calculate and measure the error, serving the subsequent risk perception.



# Environment Setup

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
