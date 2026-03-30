# Video Relighting with DiLightNet

`infer_vid.py` relights an input video under new lighting conditions. It processes each frame independently through a 4-stage pipeline and supports resuming interrupted runs.

## Usage

```bash
python infer_vid.py --input_vid <video_path> [options]
```

## Lighting Modes

```bash
# Static point light
python infer_vid.py --input_vid video.mp4 --pl_pos "0,5,3" --power 1200

# Environment map
python infer_vid.py --input_vid video.mp4 --env_map lights.hdr --env_rotation 45

# Rotating light (full loop over the video duration)
python infer_vid.py --input_vid video.mp4 --rotate_light
python infer_vid.py --input_vid video.mp4 --env_map lights.hdr --rotate_light
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--input_vid` | *required* | Input video path |
| `--out_vid` | `<input>_relit.mp4` | Output video path |
| `--prompt` | `""` | Text prompt for diffusion model |
| `--pl_pos` | `"0,5,3"` | Point light position `"x,y,z"` |
| `--power` | `1200` | Point light power |
| `--env_map` | `None` | HDR environment map path |
| `--env_rotation` | `0` | Env map rotation in degrees |
| `--rotate_light` | `False` | Rotate lighting across frames |
| `--fov` | auto | Camera FOV (estimated from first frame if omitted) |
| `--use_sam` | `True` | Use SAM for mask refinement |
| `--mask_dir` | `None` | Pre-computed masks directory (`frame000000.png`, ...) |
| `--mask_threshold` | `25.0` | Foreground extraction threshold |
| `--seed` | `3407` | Random seed |
| `--steps` | `20` | Diffusion steps |
| `--cfg` | `3.0` | Classifier-free guidance scale |
| `--inpaint` | `False` | Inpaint background (point light mode only) |
| `--cache_dir` | `tmp/vid_<name>` | Cache directory for intermediates |
| `--skip_existing` | `True` | Resume from where it left off |
| `--use_gpu_for_rendering` | `True` | GPU-accelerate Blender rendering |

## Pipeline Stages

1. **Background removal** -- extracts foreground masks for all frames
2. **Mesh reconstruction + hint rendering** -- per-frame depth estimation (DUSt3R) and Blender radiance hint rendering
3. **Relighting** -- diffusion-based relighting conditioned on radiance hints
4. **Assembly** -- composites frames into output video at original FPS

All intermediate results are cached under `--cache_dir`. To re-run a specific stage, delete its outputs and run again.
