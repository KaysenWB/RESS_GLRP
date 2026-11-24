TII / RESS Sample Data

1. processed.cpkl

Contains the first 5 batches, retaining the first 16 ship slots and all 192 time steps in each batch.
Format: list[(trajectory, adjacency)].
trajectory: float32 array [192, 16, 5]. Feature order: longitude (degrees), latitude (degrees), speed (knots), heading (degrees), and vessel length (metres).
adjacency: float64 array [192, 16, 16]. Sliced to the same ship slots; weights have not been recomputed.
The source preprocessing code specifies a 10-second interpolation interval. This format does not retain timestamps or MMSI.
Set --data_root to this file to load the sample.


2. raw_ais.csv
A real source-data subset containing the first 100 records after filtering and sorting, with all 16 columns retained.

