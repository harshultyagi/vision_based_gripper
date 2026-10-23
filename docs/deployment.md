# Deployment — Hugging Face Space

Target Space: **https://huggingface.co/spaces/btechbillu/Vission-Based-grasper** (Gradio SDK,
free CPU hardware). The Space has **not been created or published yet**: the account owner has to
do the steps below.

## What gets deployed

`python scripts/build_space.py` assembles `outputs/hf_space/` (git-ignored, ≈ 21 MB):

| Path in the Space | Source |
|---|---|
| `app.py` | small entry point that runs `app/app.py --host 0.0.0.0 --port 7860` |
| `app/app.py`, `app/app_config.yaml` | the Gradio app from this repo |
| `src/grasp_planner/` | the package (preprocessing, models, decoding) |
| `models/grconvnet_rgbd_image.pt`, `models/grconvnet_rgb_image.pt` | slim checkpoints (7.3 MB each) |
| `app/examples/pcd0101/0400/0618/0811` (r.png + d.tiff) | four **image-wise test-split** Cornell scenes as clickable examples. These are copied from the local dataset into the Space only, never into GitHub. Remove them by passing `--examples` with no ids if you prefer not to publish dataset images. |
| `requirements.txt` | **CPU-only** PyTorch 2.11.0 (`--extra-index-url https://download.pytorch.org/whl/cpu`), `opencv-python-headless`, the other dependencies, and the pinned Gradio version |
| `README.md` | Space metadata (`sdk: gradio`, `app_file: app.py`, `python_version: "3.12"`) and a short description |

The slim checkpoints come from `scripts/export_slim_checkpoint.py`. It strips the optimiser
state and verifies that outputs are identical to the full checkpoint.

## Verified locally

The built folder was run exactly as a Space would run it (`python app.py`, GPU hidden). Both
models loaded on CPU. Test scene pcd0400 with depth gave the same candidates as the GPU run
(rank 1: centre (303.4, 365.6), −25.6°, 85.9 px, score 1.173), taking about 1.8 s per request on
this laptop's CPU. Hugging Face's free CPU hardware may be slower. **Not verified:** the build on
Hugging Face's servers itself (it needs the account).

## Steps for the owner

1. Create a free account at https://huggingface.co/join with the username **btechbillu**.
2. Create a Space at https://huggingface.co/new-space: owner `btechbillu`, name
   `Vission-Based-grasper` (exact spelling; fix it consistently in `scripts/build_space.py` and the
   slides if you prefer "Vision"), SDK **Gradio**, hardware **CPU basic (free)**, public.
3. Create an access token with *write* permission: https://huggingface.co/settings/tokens
4. Upload from this repository:

```bash
.venv\Scripts\python.exe scripts\build_space.py
```

```bash
.venv\Scripts\hf.exe auth login
```

```bash
.venv\Scripts\hf.exe upload btechbillu/Vission-Based-grasper outputs\hf_space . --repo-type space
```

   (`hf` ships with the `huggingface_hub` package that Gradio already installs; verified with
   version 2.3.0. Older versions use `huggingface-cli login` and `huggingface-cli upload ...`
   with the same arguments.) Alternatively, drag the *contents* of `outputs/hf_space/` into the
   Space's **Files → Add file → Upload files** page.
5. The Space builds automatically (a few minutes: it installs PyTorch). Open the **Logs** tab if
   the build fails. The app's status panel must list both models.

## Updating

After retraining: `scripts/export_slim_checkpoint.py --run <run>` → commit `models/*.pt` →
`scripts/build_space.py` → upload again.

## Why not Vercel

Vercel runs short-lived serverless functions with a bundle-size limit, while this app is a
long-running Gradio server that needs PyTorch. A Space (or another container host such as Render
or Railway) fits better. This was not tried on Vercel.
