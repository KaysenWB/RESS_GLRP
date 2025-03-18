# SS-GLMP

This is the implementation code for the paper "Enhancing Risk Perception by Integrating Ship Interactions in Multi-Ship Encounters: A Graph-Based Learning Method".

Implementation and performance comparison of several common deep learning networks for ship motion prediction, including LSTM, GRU, Seq2Seq, TCN, Transformer, and our network GLMP. The input data are processed AIS data containing motion information and adjacency matrix, processed with reference to https://github.com/KaysenWB/AIS-Process.git.

Based on the output of the motion prediction, we present codes for DCPA and TCPA to calculate and measure the error, serving the subsequent risk perception.



# Research problem
integrating ship interactions to enhance motion prediction and risk perception performance.

![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure01.jpg?raw=true)

# Overview of this work
![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure02.jpeg?raw=true)

# Network structure
![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure03.jpg?raw=true)

# Interactive learning based on graph networks and VAE architecture
![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure04.jpg?raw=true)

# Motion prediction performance and comparison (Trajectory)
![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure08.jpeg?raw=true)

# Motion prediction performance and comparison (SOG and COG)
![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure06.jpg?raw=true)

# Risk perception performance and comparison (DCPA and TCPA)
![Figure](https://github.com/KaysenWB/SS-GLMP/blob/main/Figures/Figure10.jpeg?raw=true)
