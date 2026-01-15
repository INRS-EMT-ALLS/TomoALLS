# ALLS Betatron Computed Tomography (TomoALLS)
This is a Python library for computed tomography using the ALLS Betatron. It contains an image pre-processing pipeline for the initial tomographic scans which performs various corrections, removes streaks and aligns multiple scans. It abstracts reconstruction code for ease of use; all that needs to be done is fill out the .json file and properly named scans, and the reconstruction can be performed in a few lines.

<p align="center">
  <img src=https://raw.githubusercontent.com/INRS-EMT-ALLS/TomoALLS/refs/heads/main/images/close_up_rod.png>
</p>

# Example reconstruction

The test data was an alloy rod with an indentation. 

<p align="center">
  <img src=https://raw.githubusercontent.com/INRS-EMT-ALLS/TomoALLS/refs/heads/main/images/example_reconstruction.png>
</p>

# Author

Olivier Saint-Vincent (olivier520100@gmail.com)
