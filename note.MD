# Notes on Similarity, Loss, and Training

## Similarity Matrix Construction
- Images and texts are encoded into a shared embedding space and L2-normalized so that their dot product becomes cosine similarity.
- Multiplying image features by the transpose of text features (scaled by the learnable temperature `logit_scale`) yields `logits_per_image`, with diagonal entries representing matched pairs and off-diagonals denoting mismatches.

## Contrastive Loss Function
- Training treats each image row of `logits_per_image` as a classification task over captions and each column (or `logits_per_text`) as a classification task over images.
- Applying cross-entropy in both directions maximizes diagonal similarities while minimizing off-diagonal similarities, reinforcing true matches and discouraging mismatches.

## Training Step Highlights
- The training loop encodes batches of preprocessed images and tokenized texts, normalizes the resulting embeddings, and computes the similarity matrices.
- Symmetric cross-entropy losses are computed, gradients are backpropagated, and optimizer updates adjust both the model weights and the temperature parameter, aligning the embedding space for image-text pairs.
