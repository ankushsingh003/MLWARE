
The task is to reconstruct the chronological order of scrambled frames from corrupted short video clips. Re-ordering the frames uncovers the physics, temporal continuity, and cause-effect relationships of the contained motions. Our problem is effectively translated to sequence ranking, where each image must be assigned a rank indicating its chronological position among the set of frames.

Our solution employs a hybrid deep learning model (`FrameReorderingModel`), combining a ResNet-18 spatial feature extractor with a Transformer Encoder.
- **Feature Extractor:** We utilize a pretrained ResNet-18 model to process each scrambled frame independently, producing 512-dimensional abstract feature embeddings.
- **Sequence Modeling:** A multi-head self-attention Transformer Encoder takes the set of unordered frame embeddings and captures global context, recognizing visual similarities and potential motion trails between different frame pairs.
- **Prediction Head:** A fully connected Multi-Layer Perceptron (MLP) calculates a scalar score for each transformed frame embedding. Sorting these scores from lowest to highest produces the final predicted ordering of the frame sequence. 

We apply a resizing function and standard ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) to prepare the data for the ResNet architecture. This avoids massive video memory consumption but preserves the visual content of the frames. Data is organized via a custom PyTorch `Dataset`.

The objective is to train the scoring head using a **Pairwise Margin Ranking Loss**. By exhaustively comparing the predicted scores for all possible intra-video frame pairs against the ground truth labels, the loss function strongly penalizes chronological violations.
For evaluation, we employ the primary competitive metric: **Kendall's Tau Correlation Coefficient**, counting concordant and discordant chronological pairs as per the exact calculation rules from the competition problem statement.

By framing the unscrambling problem as an embed-and-score sequence ranking task rather than relying heavily on hard-coded heuristics or frame-to-frame optical flow, the model maintains a high degree of efficiency while ensuring strict differentiation between physical events happening at opposite ends of the video timeline.
