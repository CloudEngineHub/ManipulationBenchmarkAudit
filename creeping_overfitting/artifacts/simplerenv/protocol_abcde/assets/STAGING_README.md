# Custom Asset Staging Note

## Purpose

This note explains how the accepted custom SimplerEnv assets in this package relate to runtime trees. It is a staging note only; it does not modify any third-party checkout.

The files under `assets/custom/` are the accepted custom asset files for this staging package. See [THIRD_PARTY_NOTICES.txt](../../../../../THIRD_PARTY_NOTICES.txt) for the upstream notice covering the included SimplerEnv collision mesh and asset-manifest entries.

Accepted asset IDs included here:

- `render_candidate_blue_hybrid_v4`
- `render_candidate_red_corrected_v6e`
- `render_candidate_white_offwhite_hybrid_v4`

For reproduction, stage these files into each runtime tree that evaluates Protocol C/D/E so that the runtime has:

- `ManiSkill2_real2sim/data/custom/info_bridge_custom_baked_tex_v0.json`
- `ManiSkill2_real2sim/data/custom/models/render_candidate_blue_hybrid_v4/`
- `ManiSkill2_real2sim/data/custom/models/render_candidate_red_corrected_v6e/`
- `ManiSkill2_real2sim/data/custom/models/render_candidate_white_offwhite_hybrid_v4/`

Do not treat this package as an instruction to edit `third_party/` in place. Stage the files into the intended runtime tree explicitly and verify the runtime asset manifest/preflight before launching rollouts.
