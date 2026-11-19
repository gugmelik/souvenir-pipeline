# Assets (made once per product, reviewed by a person)

| Folder | What goes here | How to make it |
|---|---|---|
| `style_refs/` | Images that define the look, named `slotzilla_style_01.png`, `_02`, ... (the first is the neon cartoon Fremont scene) | The spec refers to them by glob `slotzilla_style_*.png`; the first one is passed to every stylize step. Use images you own; **not** frames from studio films. A picture with a character in this style (not a real guest) helps faces most. |
| `style_refs/slotzilla_character_01.png` | **A cartoon sample person** in the wanted style: how riders' faces and bodies are drawn. Used by the stylize step instead of the scene image when present. | One full-body character on a plain or simple background works best. It must not be a real guest. Qwen can copy hair or face from it, so if every guest starts to look alike, use a more generic character. |
| `poses/` | Pose drawings for each zipline pose (`prone_superman.png`, `prone_celebrate.png`, `prone_point.png`) | `python -m souvenir poses` draws simple colored skeletons. Better: photos of a staff member in the real harness, on a plain background. |
| `plates/candidates/` | Draft background plates (scene with no riders) | `python -m souvenir plates --n 8` (uses the style ref + `plate_prompt` from the product spec). Files starting with `REJECT_` had banned text detected by OCR. |
| `plates/approved/` | The plates actually used, named `slotzilla_*.png` (the ComfyUI plate workflow loads `slotzilla_plate_01.png`) | Copy the best candidates here **after a person checks them**: correct "Las Vegas 2026" spelling, no third-party brands or logos, the client is happy with how their tower and sign look. |

Rules of thumb
- Plates are 3:4 portrait (1200x1600) with empty sky where riders will go; cables are drawn by the compositor, so plates can have none.
- Keep 5-10 approved plates per product so families get some variety (the planner picks one per run with the run seed).
- Everything in here is versioned with the product spec: when you change the style refs, re-run `souvenir eval` before shipping.
