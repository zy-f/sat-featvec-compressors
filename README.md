# Satellite Feature Vector Compressors
### Author: Christopher Polzak (zy-f)

EE269 (Winter 2023) final project.

Experiments on compression methods for satellite feature vectors.

[Paper](EE269_FinalProjectReport.pdf) | [Poster](EE269_Final_Poster-fixed.pdf)

## Summary
In this project, we investigate what types of compression are most effective at taking the satellite image features extracted from the Geography-Aware SSL model (a deep feature extraction neural network) and further reducing the representations into quantized and/or lower dimensional space while preserving downstream performance when using the features.

Through our experiments, we find that:
- Simple compression schemes such as PCA and naive uniform quantization with clipping are more than sufficient to keep up with and even surpass the performance of neural network-based compression schemes.
- Uniform quantization from 32-bit float to 8-bit integer is effectively lossless
- PCA and uniform quantization (as compared to NN compressors) have greater gains in performance when using deeper downstream models, suggesting there is more latent information to be tapped into
