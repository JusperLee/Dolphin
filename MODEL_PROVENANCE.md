# Dolphin Model Provenance and Use Notice

_Last updated: July 22, 2026_

This document describes the provenance of the model artifacts currently
published at [JusperLee/Dolphin](https://huggingface.co/JusperLee/Dolphin).
It is separate from the licensing of Dolphin source code.

This document is a provenance disclosure, not a legal conclusion about the
copyright status of trained or distilled model parameters.

## License scope

The Apache License 2.0 in this repository applies to Dolphin-owned source code
and to compatible incorporated source code described in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

That source-code license does not, by itself, grant rights in:

- released model weights;
- training datasets or source recordings;
- AV-HuBERT software, checkpoints, or outputs;
- names, likenesses, voices, performances, or other third-party material.

## Current released checkpoint

The current Hugging Face release contains `model.safetensors`.

Its published training provenance is:

| Component | Published provenance |
|---|---|
| Video encoder | DP-LipCoder |
| Video-encoder inputs | LRS2, LRS3, and VoxCeleb2 mouth data, according to the public pretraining recipe |
| Distillation teacher | AV-HuBERT, using the `large_vox_433h.pt` checkpoint |
| Teacher-target generation | `videoencoder_pretrain/extract_avhubert_mouth_features.py` |
| Published distillation configuration | AV-HuBERT target embeddings with `distill_cost: 1.0` in `configs/videoencoder_pretrain.yml` |
| Separator training data | VoxCeleb2 mixtures, according to the current model card |

The AV-HuBERT checkpoint itself is not bundled in the Dolphin GitHub or
Hugging Face repository. Its feature outputs were nevertheless used as
training targets for the video encoder contained in the currently released
checkpoint.

## AV-HuBERT notice

AV-HuBERT is distributed by Meta under the custom
[AV-HuBERT License Agreement](https://github.com/facebookresearch/av_hubert/blob/main/LICENSE),
rather than Apache-2.0.

The upstream attribution is:

> AV-HuBERT is licensed under the AV-HuBERT license, Copyright (c) Meta Platforms, Inc. All Rights Reserved.

No separate permission from Meta is included with this Dolphin release.
The Dolphin authors do not take a definitive position here on whether the
resulting distilled parameters constitute a derivative work or otherwise fall
within particular upstream terms.

Pending written clarification of those terms, we do not represent the current
checkpoint as cleared for commercial or production use. Users should not
interpret the Apache-2.0 source-code license or the availability of the model
artifact as such authorization.

## Training-data notice

The public training procedure references LRS2, LRS3, and VoxCeleb2. Dolphin
does not redistribute the original datasets through this repository. Those
datasets, their source media, and associated personal or performance rights
remain subject to their own terms and applicable law.

No rights in those datasets or their underlying recordings are granted by the
Dolphin source-code license. Availability of a trained checkpoint does not
constitute a representation that a particular downstream use has been cleared
by every relevant data or media rightsholder.

Exact dataset editions, manifests, and split hashes are not presently included
with the public checkpoint. Users requiring auditable provenance should treat
this as an unresolved item.

## AV-HuBERT-free checkpoint status

No currently published Dolphin checkpoint is identified as having been trained
without AV-HuBERT code, checkpoints, and feature outputs. No controlled,
apples-to-apples performance result for that exact configuration has been
published.

Any future checkpoint represented as AV-HuBERT-free should:

1. be trained from a fresh initialization without loading AV-HuBERT code,
   checkpoints, or generated targets;
2. remove the encoded AV-HuBERT target input, rather than only setting
   `distill_cost=0`;
3. use separately reviewed training data;
4. receive a distinct model version and its own provenance statement.

## Questions

For provenance questions, open a GitHub issue or contact
[tsinghua.kaili@gmail.com](mailto:tsinghua.kaili@gmail.com).
