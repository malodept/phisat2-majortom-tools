# Alignment notes

This repository does not yet reproduce the full LightGlue triplet generation pipeline.

For now, the package builder assumes that corrected full-scene PhiSat-2 georeferencing information is already available. In the demo, this corrected geometry is estimated from georeferenced PhiSat-2 / Sentinel-2 patches derived from:

https://huggingface.co/datasets/lorenzopapa53/phisat2-s2-lightglue-triplets

The current method can be summarized as:

1. use LightGlue-georeferenced PhiSat-2/Sentinel-2 patch pairs;
2. recover full-scene pixel-to-world control points;
3. fit a full-scene affine model;
4. apply the known horizontal flip correction;
5. warp the raw PhiSat-2 L1C acquisition to the official MajorTOM 10 m cell grid.
