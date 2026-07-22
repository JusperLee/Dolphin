# Third-Party Notices

Dolphin includes code adapted from the open-source projects listed below.
The original code and Dolphin's modifications are distributed under the
Apache License, Version 2.0. The complete license text is provided in
[`LICENSE`](LICENSE).

Copyright in the original code remains with its respective copyright
holders. This notice is informational and does not modify any applicable
license terms.

## IIANet

- Project: [JusperLee/IIANet](https://github.com/JusperLee/IIANet)
- License: [Apache License 2.0](https://github.com/JusperLee/IIANet/blob/main/LICENSE)
- Relationship to Dolphin: portions of Dolphin's model, training, or
  evaluation implementation were adapted from IIANet and modified for the
  Dolphin architecture and release.

## SepReformer

- Project: [dmlguq456/SepReformer](https://github.com/dmlguq456/SepReformer)
- License: [Apache License 2.0](https://github.com/dmlguq456/SepReformer/blob/main/LICENSE)
- Relationship to Dolphin: portions of Dolphin's speech-separation
  implementation were adapted from SepReformer and modified for the Dolphin
  architecture and release.

## Other dependencies

Packages installed through `requirements.txt`, `package.json`, or other
environment-management files are not relicensed by Dolphin. Each dependency
remains subject to its own license and notices.

AV-HuBERT was used in the training provenance of the currently released model
checkpoint but is not listed here as Apache-licensed Dolphin source code. See
[`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md) for that disclosure.
