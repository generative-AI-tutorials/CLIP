# Notes on Similarity, Loss, and Training

## Similarity Matrix Construction
- Images and texts are encoded into a shared embedding space and L2-normalized so that their dot product becomes cosine similarity.
- Multiplying image features by the transpose of text features (scaled by the learnable temperature `logit_scale`) yields `logits_per_image`, with diagonal entries representing matched pairs and off-diagonals denoting mismatches.
- The same matrix is transposed to obtain `logits_per_text`, making the image–text interaction symmetric and allowing the loss to view each modality as the classifier over the other.

## Temperature Scaling and Normalization
- Normalization constrains feature vectors to the unit sphere, so logits reflect angular distance rather than raw magnitude.
- The learnable temperature `logit_scale.exp()` rescales cosine similarities to sharpen or soften the softmax distribution. During training, gradients encourage higher temperatures (up to a capped value) when classes are hard to separate and lower temperatures when similarities are already peaky.
- In the implementation, `logit_scale` starts at `log(1/0.07)`, matching the initialization from the CLIP paper; it is stored as an unconstrained scalar parameter and exponentiated in the forward pass before scaling the similarity matrix. This keeps the learnable quantity in log-space while ensuring the applied temperature is always positive.
- The exact initialization lives in `clip/model.py` inside the `CLIP` constructor: `self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))`, and the forward pass multiplies the normalized similarity matrix by `self.logit_scale.exp()` (`logits_per_image = logit_scale * image_features @ text_features.t()`).

## Contrastive Loss Function
- Training treats each image row of `logits_per_image` as a classification task over captions and each column (or `logits_per_text`) as a classification task over images.
- Targets are simple identity indices (e.g., image *i* should match text *i*), so cross-entropy pushes the diagonal logits toward 1-hot dominance while suppressing off-diagonals.
- Applying cross-entropy in both directions maximizes diagonal similarities while minimizing off-diagonal similarities, reinforcing true matches and discouraging mismatches.
- For a single image row, `logits_per_image[i]` becomes the input to `softmax`; the loss term for that image is `-log softmax(logits_per_image[i])[i]`, which is low only when the *i*-th caption logit is the largest by a comfortable margin.

## How Cross-Entropy Connects to Similarity
- Cross-entropy computes `-log softmax(logits)`, so each matched pair is rewarded when its scaled cosine similarity is larger than all mismatched pairs in the same row or column.
- Because the logits are cosine similarities multiplied by `logit_scale`, improving the loss directly drives image–text cosine distances closer together for true pairs and farther apart for negatives.
- The shared temperature means both `logits_per_image` and `logits_per_text` adapt together, encouraging a balanced embedding geometry where neither modality dominates.

## Training Step Highlights
- The training loop encodes batches of preprocessed images and tokenized texts, normalizes the resulting embeddings, and computes the similarity matrices.
- Symmetric cross-entropy losses are computed, gradients are backpropagated, and optimizer updates adjust both the model weights and the temperature parameter, aligning the embedding space for image-text pairs.
