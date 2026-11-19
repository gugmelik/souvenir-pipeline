# occo souvenir pipeline

Turns a guest photo into a stylized souvenir (first product: **Slotzilla "I'M FLYING"**) by splitting
one long prompt into short, checkable steps. Open editing models such as Qwen-Image-Edit-2511 do one
edit well and four edits at once badly; this project gives them one edit at a time and moves all the
logic (counting people, choosing a formation, banned brands, retries) into code.

```
guest photo
   |
   v
1 ANALYZER   detector counts N riders (code)  ->  crops per person  ->  VLM "person card"
   |         (clothing, hair, accessories - never race/ethnicity)
   v
2 PLANNER    code: formation, slots, plate, seeds, which images each step gets
   |         LLM: writes each step's prompt (<=120 tokens, no brands, no negations), validated,
   |         template fallback
   v
3 EXECUTOR   per person:  pose (photo + pose drawing)  ->  stylize (+ house style + face crop)
   |         ->  cutout          each step is checked and retried with a new seed if it fails
   v
4 COMPOSER   N cutouts placed on an APPROVED background plate, one cable each (plain code)
   |
   v
5 PASSES     harmonize (light blend)  = merch.png  (clean, no motion lines)
   |         effects (speed lines)    = hero.png
   v
6 VERIFIER   rider count, identity per rider, clothing per rider, product checklist, OCR text
             -> report.json
```

## Folder layout

| Path | What |
|---|---|
| `configs/pipeline.yaml` | Which image model, VLM, detector, identity scorer, segmenter; retries; thresholds |
| `configs/products/slotzilla_flight_v1.json` | The product spec: your long brief turned into data (templates, formations, poses, checklist, banned text) |
| `briefs/slotzilla_flight_brief.md` | Your original long prompt, used as the single-prompt baseline |
| `prompts/` | System prompts for the VLM: person card, planner, checklist, count, identity judge |
| `assets/` | House-style refs, pose drawings, background plates (see `assets/README.md`) |
| `src/souvenir/` | The pipeline (one module per stage, backends in `backends/`) |
| `tests/` | Offline tests (no GPU / keys) |
| `scripts/` | `setup.ps1`, `run_slotzilla_example.ps1` |
| `runs/` | Output folders (git-ignored) |

## 1. Quick start (offline, 5 minutes)

```powershell
cd C:\dev\occo\occo-souvenir-pipeline
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1     # venv, install, poses, tests, demo
.\.venv\Scripts\Activate.ps1
python -m souvenir demo --people 4
```

The demo uses a synthetic group photo and mock models, so the images are placeholders. It shows the
real structure: open `runs\demo\runs\<id>\` and look at `plan.json`, `steps\`, `composite.png`, `report.json`.

## 2. Connect the real models

All settings are in `configs/pipeline.yaml`; secrets and URLs go in `.env` (copy of `.env.example`).

**Image model: Qwen-Image-Edit-2511 via `gpu-instances`** (`C:\dev\occo\gpu-instances`, the default)

It runs Qwen-Image-Edit-2511 with the Lightning 4-step LoRA and takes up to 4 images per call, so the
stylize step gets all 3 inputs. Pick how to reach it with `GPU_INSTANCES_MODE` in `.env`:

| Mode | Use when | How it connects |
|---|---|---|
| `pod` (default) | Development: one pod started by hand | In gpu-instances run `scripts\start.bat`. The pipeline reads the URL and key from `gpu-instances\.state.json` and calls `POST /v1/images/edit`. Run `scripts\stop.bat` there when done. |
| `gateway` | Batches / production | Production job API: uploads inputs (`POST /v1/uploads`), submits `POST /v1/jobs` with an Idempotency-Key, polls `GET /v1/jobs/{id}`. Uses `PROD_URL` / `PROD_API_KEY` from `gpu-instances\.env`. The fleet manager starts and stops pods for you. |

Settings are in `configs/pipeline.yaml -> backend.gpu_instances` (4 steps, true_cfg 1.0, max 4 images,
max side 2048, matching the server defaults). Other backends (`genai_instance`, `modal_comfy`,
`diffusers`, `mock`) remain available with `SOUVENIR_BACKEND`.

**VLM** (analyzer, planner, verifier): any OpenAI-compatible server.
Recommended: self-host Qwen3-VL with vLLM on a RunPod GPU, so guest faces stay in your infrastructure:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 16384
```

Then set `VLM_BASE_URL=http://<pod>:8000/v1`. A hosted API also works for testing; check its data
terms before sending guest photos. `planner.mode: template` runs without any LLM for the planner.

**Detector** (`SOUVENIR_DETECTOR`): `insightface` (buffalo_l) or `rekognition` (what production uses;
also gives person boxes). Faces smaller than 35% of the largest face are treated as bystanders
(`detector.min_face_ratio`).

**Identity** (`SOUVENIR_IDENTITY`): `vlm_judge` (default, handles cartoon faces best), `insightface`
(good for photoreal steps, weak on cartoons), `rekognition`, or a `styleid` hook for the StyleID
style-robust encoder.

**Segmenter**: `rembg` with `isnet-general-use`, or `onnx_isnet` pointing at your fine-tuned ISNet export.

## 3. Prepare the product assets (once)

1. **House style refs**: put 3-10 images in `assets/style_refs/house_3d_animated_*.png`. These replace
   "Disney / Pixar style" in the brief: a named studio style is vague (every model reads it differently)
   and is an IP risk on merchandise.
2. **Pose refs**: `python -m souvenir poses` draws skeletons. Photos of a staff member in the real
   harness are better.
3. **Plates**: `python -m souvenir plates --n 8` drafts scenes with no riders into
   `assets/plates/candidates/`. A person approves them (spelling of "Las Vegas 2026", no third-party
   logos, client OK with their tower and sign) and copies them to `assets/plates/approved/slotzilla_*.png`.
   The plates can also come from any tool or an artist. Generating them once means brand safety and
   text are checked once, not in every guest's prompt.

## 4. Run

```powershell
# analyze + plan only: cheap way to check the detector, person cards and prompts
python -m souvenir plan --image "https://assets.occo.io/...jpg"

# full pipeline
python -m souvenir run --image "https://assets.occo.io/...jpg" --seed 7

# baseline: your original long brief in ONE Qwen call (for comparison)
python -m souvenir run --image photo.jpg --mode single:main --brief briefs\slotzilla_flight_brief.md

# how long is the brief, and where does the 512-token default cut it?
python -m souvenir tokens --file briefs\slotzilla_flight_brief.md
```

`scripts\run_slotzilla_example.ps1` runs all four on your Slotzilla example photo.
Use `--mock-backend` to test the detector + VLM + planner + compositor without spending GPU time,
and `--set key=value` to override any config value (for example `--set planner.mode=template`).

### Run it in ComfyUI

**Full pipeline as a ComfyUI workflow, with custom nodes:** see [`comfyui/README.md`](comfyui/README.md)
(`comfyui\workflows\occo_souvenir_slotzilla.json`, install with `comfyui\install.ps1`).

**Core-nodes-only export for the Vast endpoint:**

`souvenir comfy` runs the analyzer and planner on one photo. It then writes the whole plan as **one ComfyUI
API-format workflow**, so you can look at the result before standing up the pipeline. It uses the same core
nodes and model files as `C:\dev\occo\comfyui-api\workflows\qwen_image_edit_2511.json` (Qwen-Image-Edit-2511
fp8mixed + Lightning 4-step).

```powershell
# Vast endpoint via your comfyui-api project: uploads the inputs, sends the workflow, saves the outputs
python -m souvenir comfy --image "https://assets.occo.io/...jpg" --run

# or only write the files: then drag workflow_api.json into a local ComfyUI after copying inputs\* to ComfyUI\input
python -m souvenir comfy --image photo.jpg
```

Every stage has its own SaveImage (`plate`, `person_<i>_pose`, `person_<i>_style`, `compose_<k>`, `merch`,
`hero`), so you can see which step goes wrong. Things that differ from the deployed pipeline:
- **Compositing:** riders are placed by Qwen, two per pass, using position words from the formation
  ("on the right, low, large and close to the camera"). The pipeline instead pastes cutouts at exact
  coordinates. This makes the preview the "group call" A/B variant.
- **No checks or retries.** Change `--seed` to try again.
- **Missing assets:** without an approved plate, the workflow drafts one from a blank canvas. Without a
  house-style image, stylize uses the text style description only.
- **Analyzer and planner:** they still need a detector (InsightFace on your machine is fine) and, ideally,
  the VLM. Add `--set vlm.kind=mock --set planner.mode=template` to run without a VLM, with generic
  person descriptions.

A photo with N people needs 2N + ceil(N/2) + 1 Qwen edits (+1 for a drafted plate). That's 9 for 3 people,
all in one ComfyUI run.

### What is in a run folder

```
runs/<date>_<photo>_pipeline/
  source.png, analysis.json        N, boxes, person cards, ignored bystanders
  crops/                           person_<i>_body.png, person_<i>_face.png
  plan.json                        formation, slots, every step's prompt + seed, planner warnings
  steps/                           person_<i>_pose_a<k>.png, ..._stylize_a<k>.png, cutouts, harmonize, effects
  composite.png, layout.json       riders on the plate + each rider's box
  merch.png                        harmonized, no motion lines (mugs, t-shirts)
  hero.png                         with speed lines (the main souvenir)
  report.json                      checks per output, retries, backend calls, seconds, warnings
```

## 5. Evaluate against the single prompt

```powershell
python -m souvenir eval --images C:\data\slotzilla_test --modes pipeline,single:main --limit 50
# add single:gemini after enabling baselines.gemini in pipeline.yaml and setting GEMINI_API_KEY
```

This writes `runs\eval\results.csv`, `summary.md` (pass rate, count accuracy, identity, calls and
seconds **by number of people**) and `index.html` (side-by-side sheet for human review). Use 50+ real
photos with 1 to 6 people. Group size is where single-prompt editing breaks.

## 6. Calibrate the identity threshold

Thresholds in `pipeline.yaml -> identity.thresholds` are starting points. Per style:
1. Run 30-50 photos, then label in `index.html` which riders are recognizable (yes/no).
2. Pick the threshold where the scorer best separates yes from no (ROC / AUC).
3. Photo-trained face models (InsightFace, Rekognition) score cartoons low even when people
   recognize them. Judge them by ranking (AUC), not by the raw number, or use `vlm_judge` / StyleID.

## 7. Add a new product

Copy `configs/products/slotzilla_flight_v1.json`, change the templates, poses, formations, plate
prompt, checklist and banned terms, add the assets, then run
`python -m souvenir run --product configs/products/<new>.json ...`. For a non-flying product
(boat cabin, roller coaster), set `"cable_direction": "none"`.

## Known limits and next steps

- **Person isolation**: body crops are clipped between neighbours' faces. In tight hugs, arms from a
  neighbour can leak in. Next upgrade: SAM 2 prompted with each face box.
- **Group call A/B**: for 2-3 people, also try plate + all stylized riders in one Qwen call
  (2511 is better at groups); keep compositing as the fallback for big groups.
- **Distillation**: runs that pass every check are training pairs (photo + full instruction -> hero).
  Train a LoRA on Qwen-Image-Edit-2511 or FLUX.2 [klein] 4B so one call does what the pipeline does.
- **RL**: then FlowGRPO with reward = identity + VLM checklist + aesthetics.
- **Cost**: N riders = 2N + 2 image calls (8 for 3 people), more with retries. `scripts\bench.bat` in
  gpu-instances gives $ per image, so $ per souvenir is about (2N + 2) x that.

## Licences and legal checklist (confirm with a lawyer before production)

- **Guest photos**: these are biometric data, often of children. Training on them needs a clear legal
  basis and consent beyond what photo delivery needs. Keep processing in occo's AWS regions.
- **InsightFace pretrained weights** (`buffalo_l`): non-commercial research only. Use Rekognition or a
  commercially licensed model in production.
- **Qwen-Image-Edit-2511**: Apache 2.0. **Qwen-Image-2.1**: research licence only.
  **FLUX.2 [klein] 4B**: Apache 2.0; FLUX.2 [dev] needs a BFL licence to run commercially.
- **Gemini (Nano Banana)**: fine as a comparison baseline. Its terms prohibit using the service to build
  competing models, so do not train on its outputs without a legal review.
- **Style**: no studio names in prompts (`banned_terms_in_prompts`). Use your own or licensed
  house-style references.
- **Brands**: third-party brand text is checked by OCR (`banned_text_ocr`). The client's own SLOTZILLA
  sign is in the plate. Confirm the client approves it on merchandise.
