# Release licensing review

Checked 30 October 2026. The publisher authorized public distribution and permissive reuse of the weights after stating that a permission or reviewed basis was available. No permission document was supplied or independently reviewed in this repository.

| Material | Source terms | Release treatment |
|---|---|---|
| Reproduction code | MIT, copyright ETH Zurich / Chong Zhang | Retain [LICENSE](../LICENSE) and attribution |
| Original LAFAN1 motions | [CC BY-NC-ND 4.0](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt) | Do not bundle CSV/BVH motion data |
| Retargeted motion files | [Dataset card](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) retains the original LAFAN1 terms | Do not treat the repository's Unitree BSD notice as a replacement for the motion terms |
| G1 robot description | Unitree BSD-3-Clause notice in the dataset repository | Download separately, do not bundle meshes or URDFs |
| Our policy weights | MIT for the publisher's rights in the weights | Permissive publisher grant. It does not relicense third-party motion content |
| New motion videos/GIFs | Rendered policy and retargeted LAFAN1 reference motions | Retain motion provenance and CC BY-NC-ND 4.0 terms |
| Evaluation numbers and new plots | Our measured results and presentation | Include with protocol and attribution |

The dataset card explicitly distinguishes the motion license from its code license. [Creative Commons' AI guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/) explains that whether a model or output is an adaptation depends on the actual use and applicable rights. Its conservative approach does not treat ND training data as generally cleared for model release.

The publisher's authorization is recorded separately from the source-license facts. The MIT grant covers the publisher's rights only. It does not change LAFAN1's terms or independently settle which uses involve third-party rights. The release does not claim a permissions document was reviewed. No affiliation or endorsement by Ubisoft, Unitree, ETH Zurich or the original researchers is implied.
