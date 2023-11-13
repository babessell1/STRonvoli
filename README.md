# STRonvoli
CNN for enhancing short read STR calls

Pre-processing steps: 
- One-hot encode bases at each position
- Custom Dataset, Dataloader for base, depth, metadata 

3 methods:
1) Baseline CNN -> FCN with concatenated metadata
2) Dilated self-designed kernel in CNN -> FCN w/concatenated metadata
3) 2 convolutional networks with regular kernel -> aggregate -> FCN w/concatenate metadata

We are maintaining same number of convolutional/FCN layers and number of kernels for the 3 models so they aren't confounding.
