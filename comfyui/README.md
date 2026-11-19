# The souvenir pipeline as a ComfyUI workflow

`workflows/occo_souvenir_slotzilla.json` runs the whole pipeline inside ComfyUI, from guest photo to
hero image. Core ComfyUI has no nodes for some steps, so `ComfyUI-occo-souvenir/` adds them.

| Node | What it does | Why it is custom |
|---|---|---|
| Occo Product Spec | loads `configs/products/slotzilla_flight_v1.json`: templates, poses, checklist; optional house-style image | product data |
| Occo Detect People | counts riders in code, crops body + face per person, person cards (optional VLM) | detection + cropping |
| Occo Plan | formation, slots, seeds, every step prompt (template or LLM planner) | planning |
| Occo Qwen Edit Riders | for each person: pose -> stylize with Qwen-Image-Edit-2511 | the number of people varies and ComfyUI has no loops |
| Occo Cutout | removes each rider's background (rembg if installed, else white-background key) | |
| Occo Compose (exact layout) | pastes riders on the plate at the formation's positions, one zipline cable each | geometry |
| Occo Verify | rider count + identity per rider -> text report | checks |

Everything else uses **core nodes**: the model loaders, the plate draft, harmonize (= merch) and effects
(= hero). They are the same nodes and model files as `C:\dev\occo\comfyui-api\workflows\qwen_image_edit_2511.json`.

The workflow is grouped into bands:

1. **Guest photo -> people -> plan**: rider crops and every prompt show up in preview nodes.
2. **Background plate**: drafted from a blank canvas + the spec's plate prompt.
3. **Riders**: posed, stylized, cutouts, composite.
4. **Harmonize (merch) and effects (hero)**.
5. **Checks**: the report as text.

## Install on Linux (after `git clone`)

```bash
source ~/ComfyUI/.venv/bin/activate            # the Python that runs ComfyUI
bash comfyui/install.sh ~/ComfyUI --models bf16  # installs the package + nodes, downloads bf16 models, self-tests
# --models fp8 for the smaller files, --models none to skip downloads; --no-face / --no-rembg to skip those packages
```

Then restart ComfyUI and open `comfyui/workflows/occo_souvenir_slotzilla_bf16.json`.
The pipeline code is installed editable (`pip install -e`), and the node folder is copied into `custom_nodes`.
After `git pull`, re-run `bash comfyui/install.sh ~/ComfyUI --models none` and restart ComfyUI.
The script ends with a self-test that loads the nodes the way ComfyUI does. It prints `OK` or the exact import error.

## Install on Windows

```powershell
cd C:\dev\occo\occo-souvenir-pipeline
powershell -ExecutionPolicy Bypass -File comfyui\install.ps1 -ComfyUI "D:\ComfyUI" -WithFace -WithRembg
```

This links `ComfyUI-occo-souvenir` into `custom_nodes` (a junction, so the nodes use this project's `src\`).
It also installs the Python packages into ComfyUI's Python and lists missing model files.

Model files go in your ComfyUI `models\` folder, with the same names as in your Modal app:

| File | Folder | Hugging Face repo |
|---|---|---|
| `qwen_image_edit_2511_bf16.safetensors` or `qwen_image_edit_2511_fp8mixed.safetensors` | `diffusion_models` | Comfy-Org/Qwen-Image-Edit_ComfyUI (`split_files/diffusion_models/`) |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `text_encoders` | Comfy-Org/HunyuanVideo_1.5_repackaged (`split_files/text_encoders/`) |
| `qwen_image_vae.safetensors` | `vae` | Comfy-Org/Qwen-Image_ComfyUI (`split_files/vae/`) |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | `loras` | lightx2v/Qwen-Image-Edit-2511-Lightning |

Two versions of the workflow, identical except for the diffusion-model file:

| Workflow | Diffusion model | Size | GPU |
|---|---|---|---|
| `occo_souvenir_slotzilla_bf16.json` | `qwen_image_edit_2511_bf16.safetensors` | ~41 GB | 48 GB+ (or ComfyUI offloading with plenty of system RAM) |
| `occo_souvenir_slotzilla.json` | `qwen_image_edit_2511_fp8mixed.safetensors` | ~20 GB | 24 GB+ |

Both use the fp8 text encoder, the VAE and the Lightning LoRA (bf16) listed above. For a bf16 text encoder too,
run `python comfyui/build_workflow.py` after changing `build_api(precision, text_encoder)` in the variants list
(the file is `qwen_2.5_vl_7b.safetensors` from Comfy-Org/Qwen-Image_ComfyUI, `split_files/text_encoders/`).

## Run

1. Restart ComfyUI, then **Workflow -> Open** (or drag in) `comfyui\workflows\occo_souvenir_slotzilla.json`.
2. In **Guest photo**, upload the photo.
3. **Detect people**: `insightface` needs `-WithFace`. Without it, set `detector = manual` and type one
   face box per line (`x0,y0,x1,y1` in pixels).
4. Optional: put a VLM URL (vLLM + Qwen3-VL, OpenAI-compatible `/v1`) in `vlm_base_url` on Detect
   People / Plan / Verify. Clothing then gets described per person and the checklist runs. Without it,
   prompts say "the same outfit".
5. Optional: connect a **Load Image** with a house-style picture to `style_image` on **Product spec**.
   Without it, stylize uses the spec's text style description.
6. Optional: to use an approved plate, replace the plate band's output with a **Load Image** wired to
   `plate` on **Compose**.
7. **Run**. Outputs land in `output\souvenir\`: `plate`, `composite`, `merch`, `hero`. Rider crops, posed
   and stylized riders, cutouts, the plan and the check report show in preview nodes.

A photo with N people runs 2N + 3 Qwen edits (plate, 2 per rider, harmonize, effects): 9 for 3 people.
Change **seed** on **Plan** (or on the core KSamplers) to try variations.

## Using your own plate and style picture

1. **Make the plate once:** open `workflows/plate_cleanup_bf16.json` (or `plate_cleanup.json` for fp8). Upload
   an example souvenir you like as **Example souvenir**, then Run. It runs two short edits:
   - remove the rider, harness, cable, speed lines and WOOO! text;
   - fix the sign text (SLOTZILLA spelling, generic words on the other signs).

   Both steps are saved (`souvenir/plate_cleanup_step1`, `souvenir/slotzilla_plate`). Check the result and
   re-run with another seed if something is left over.
2. **Use it:** open `workflows/occo_souvenir_slotzilla_bf16_plate.json`. It has two extra **Load Image** nodes:
   - **Background plate**: the cleaned plate.
   - **Style reference**: an image in the look you want for the riders, for example the original souvenir. If
     guests start getting that rider's hair or face, use a character picture of a different person instead.
3. **Compose** takes the canvas from the plate (`canvas = from plate`, long side 1600). A landscape plate stays
   landscape and nothing gets cropped.

## Where it can run

- **Local ComfyUI, or a ComfyUI pod you control** (RunPod / Vast instance): yes, install the nodes as above.
- **comfyui-api's Vast Serverless template**: no. That template has no custom nodes, and v1 uses
  no custom image. For that endpoint use `python -m souvenir comfy --run`: core nodes only, with
  inputs prepared on your machine.
- **gpu-instances**: that server runs diffusers, not ComfyUI. The Python pipeline uses it directly.

## Files

- `ComfyUI-occo-souvenir/`: the custom nodes (`__init__.py`, `nodes.py`).
- `workflows/occo_souvenir_slotzilla_bf16.json` / `occo_souvenir_slotzilla.json` (fp8): editor format, open in ComfyUI.
- `workflows/*_api.json`: the same in API format for `POST /prompt`.
- `install.sh` (Linux) / `install.ps1` (Windows): node install.
- `build_workflow.py`: regenerates both. Run `python comfyui\build_workflow.py --comfy http://127.0.0.1:8188`
  with ComfyUI running to refresh node definitions, or without `--comfy` to use `object_info_cache.json`.
