# Release licensing review

Checked 30 October 2026. The publisher authorized public distribution after stating that a permission or reviewed basis was available. No permission document was supplied or independently reviewed in this repository.

| Material | Source terms | Release treatment |
|---|---|---|
| Reproduction code | MIT, copyright ETH Zurich / Chong Zhang | Retain [LICENSE](../LICENSE) and attribution |
| Original LAFAN1 motions | [CC BY-NC-ND 4.0](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt) | Do not bundle CSV/BVH motion data |
| Retargeted motion files | [Dataset card](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) retains the original LAFAN1 terms | Do not treat the repository's Unitree BSD notice as a replacement for the motion terms |
| G1 robot description | Unitree BSD-3-Clause notice in the dataset repository | Download separately, do not bundle meshes or URDFs |
| Our policy weights | Publisher's release: CC BY-NC-ND 4.0 | Non-commercial release, separate from the code's MIT license |
| New motion videos/GIFs | Rendered policy and retargeted LAFAN1 reference motions | Retain motion provenance and CC BY-NC-ND 4.0 terms |
| Evaluation numbers and new plots | Our measured results and presentation | Include with protocol and attribution |

The dataset card explicitly distinguishes the motion license from its code license. [Creative Commons' AI guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/) explains that whether a model or output is an adaptation depends on the actual use and applicable rights. Its conservative approach does not treat ND training data as generally cleared for model release.

The publisher's authorization is recorded separately from the source-license facts. This release does not claim that MIT automatically clears LAFAN1-derived artifacts, that other Hub uploads establish permission, or that a permissions document was independently verified. Ubisoft, Unitree, ETH Zurich and the original researchers do not endorse this fork.
