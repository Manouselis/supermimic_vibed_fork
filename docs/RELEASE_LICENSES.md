# Release licensing review

Checked 30 October 2026. This separates source notices from the unresolved release decision.

| Material | Source terms | Release treatment |
|---|---|---|
| Reproduction code | MIT, copyright ETH Zurich / Chong Zhang | Retain [LICENSE](../LICENSE) and attribution |
| Original LAFAN1 motions | [CC BY-NC-ND 4.0](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt) | Do not bundle CSV/BVH motion data |
| Retargeted motion files | [Dataset card](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) retains the original LAFAN1 terms | Do not treat the repository's Unitree BSD notice as a replacement for the motion terms |
| G1 robot description | Unitree BSD-3-Clause notice in the dataset repository | Download separately, do not bundle meshes or URDFs |
| Our policy weights | No explicit checkpoint-release grant established from the sources checked | Public release pending a resolved rights basis, no automatic MIT label |
| New motion videos/GIFs | Rendered policy and retargeted LAFAN1 reference motions | Stage locally/private only pending the same review |
| Evaluation numbers and new plots | Our measured results and presentation | Include with protocol and attribution |

The dataset card explicitly distinguishes the motion license from its code license. [Creative Commons' AI guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/) explains that whether a model or output is an adaptation depends on the actual use and applicable rights. Its conservative approach does not treat ND training data as generally cleared for model release.

This is a specific unresolved question, not a conclusion that all trained weights are prohibited. Publishing another person's checkpoint on the Hub does not establish permission for this release.

Before making the staged weights or new motion media public, record the applicable permission or reviewed basis and the chosen weight license. Do not claim Ubisoft, Unitree, ETH Zurich or the original researchers endorse this fork.
