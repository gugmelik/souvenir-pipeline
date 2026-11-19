# Assets (made once per product, reviewed by a person)

| Folder | What goes here | How to make it |
|---|---|---|
| `style_refs/` | 3-10 images that define the occo house style, named `house_3d_animated_01.png`, ... | Commission or license them, or pick approved outputs you own. **Not** frames from studio films. The spec refers to them by glob `house_3d_animated_*.png`; the first one is passed to every stylize step. |
| `poses/` | Pose drawings for each zipline pose (`prone_superman.png`, `prone_celebrate.png`, `prone_point.png`) | `python -m souvenir poses` draws simple colored skeletons. Better: photos of a staff member in the real harness, on a plain background. |
| `plates/candidates/` | Draft background plates (scene with no riders) | `python -m souvenir plates --n 8` (uses the style ref + `plate_prompt` from the product spec). Files starting with `REJECT_` had banned text detected by OCR. |
| `plates/approved/` | The plates actually used, named `slotzilla_*.png` | Copy the best candidates here **after a person checks them**: correct "Las Vegas 2026" spelling, no third-party brands or logos, the client is happy with how their tower and sign look. |

Rules of thumb
- Plates are 3:4 portrait (1200x1600) with empty sky where riders will go; cables are drawn by the compositor, so plates can have none.
- Keep 5-10 approved plates per product so families get some variety (the planner picks one per run with the run seed).
- Everything in here is versioned with the product spec: when you change the style refs, re-run `souvenir eval` before shipping.
