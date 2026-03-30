import os
from dataclasses import dataclass
from typing import Optional

import cv2
import imageio
import numpy as np
import simple_parsing
from tqdm import tqdm


@dataclass
class Args:
    input_vid: str  # Path to the input video
    prompt: str = ""  # Prompt for the generated images
    out_vid: Optional[str] = None  # Path to the output video

    seed: int = 3407  # Seed for the generation
    steps: int = 20  # Number of steps for the diffusion process
    cfg: float = 3.0  # CFG for the diffusion process

    fov: Optional[float] = (
        None  # Field of view, none for auto estimation from the first frame
    )

    use_sam: bool = True  # Use SAM for background removal
    mask_threshold: float = 25.0  # Mask threshold for foreground object extraction
    mask_dir: Optional[str] = None  # Directory of pre-computed masks (frame000.png, ...)

    # Lighting: either env_map or point light
    pl_pos: Optional[str] = (
        None  # Static point light position as "x,y,z", e.g. "0,5,3"
    )
    power: float = 1200.0  # Power of the point light
    env_map: Optional[str] = None  # Environment map path
    env_rotation: float = 0.0  # Environment map rotation in degrees (0-360)
    rotate_light: bool = (
        False  # Rotate lighting across frames (full loop over the video)
    )

    inpaint: bool = False  # Inpaint the background
    use_gpu_for_rendering: bool = True  # Use GPU for Blender rendering

    # Performance
    cache_dir: Optional[str] = None  # Cache directory for intermediate results
    skip_existing: bool = True  # Skip frames that already have outputs


def extract_frames(video_path: str) -> list[np.ndarray]:
    """Extract all frames from a video as a list of uint8 arrays (H, W, 3)."""
    frames = []
    for frame in imageio.v3.imiter(video_path, plugin="pyav"):
        frames.append(np.asarray(frame)[..., :3])
    return frames


def get_light_position(args, frame_idx: int, total_frames: int):
    """Get the point light position for a given frame."""
    if args.pl_pos is not None and not args.rotate_light:
        x, y, z = map(float, args.pl_pos.split(","))
        return [(x, y, z)]

    if args.rotate_light:
        # Rotate light around the object over the video duration
        r = 5.0
        h = 3.0
        if args.pl_pos is not None:
            parts = args.pl_pos.split(",")
            if len(parts) == 3:
                r = (float(parts[0]) ** 2 + float(parts[1]) ** 2) ** 0.5
                h = float(parts[2])
        angle = frame_idx / total_frames * np.pi * 2.0
        return [(r * np.sin(angle), r * np.cos(angle), h)]

    # Default: light from above-front
    return [(0.0, 5.0, 3.0)]


def get_env_rotation(args, frame_idx: int, total_frames: int) -> float:
    """Get environment map starting azimuth (0-1 range) for a given frame."""
    if args.rotate_light:
        return frame_idx / total_frames
    return args.env_rotation / 360.0


if __name__ == "__main__":
    args = simple_parsing.parse(Args)

    import rembg

    from demo.mesh_recon import mesh_reconstruction
    from demo.render_hints import render_bg_images, render_hint_images

    # Create rembg session once (avoids per-frame model reload → segfault)
    rembg_session = rembg.new_session(
        "sam" if args.use_sam else "u2net",
        **({"sam_model": "sam_vit_h_4b8939"} if args.use_sam else {}),
    )

    # Extract video frames
    print(f"Extracting frames from {args.input_vid}...")
    raw_frames = extract_frames(args.input_vid)
    total_frames = len(raw_frames)
    print(f"Extracted {total_frames} frames")

    # Setup cache directory
    vid_id = os.path.splitext(os.path.basename(args.input_vid))[0]
    cache_dir = args.cache_dir or f"tmp/vid_{vid_id}"
    os.makedirs(cache_dir, exist_ok=True)

    use_env_map = args.env_map is not None

    # Estimate FOV from first frame (reused for all frames for consistency)
    shared_fov = args.fov

    # --- Stage 1: Background removal for all frames ---
    print("Stage 1: Background removal...")
    masks_dir = os.path.join(cache_dir, "masks")
    frames_dir = os.path.join(cache_dir, "frames")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    for i in tqdm(range(total_frames), desc="Background removal"):
        mask_path = os.path.join(masks_dir, f"frame{i:06d}.png")
        frame_path = os.path.join(frames_dir, f"frame{i:06d}.png")

        if args.skip_existing and os.path.exists(mask_path) and os.path.exists(frame_path):
            continue

        frame = cv2.resize(raw_frames[i], (512, 512))
        imageio.imwrite(frame_path, frame)

        if args.mask_dir:
            mask = imageio.v3.imread(os.path.join(args.mask_dir, f"frame{i:06d}.png"))
            if mask.ndim == 3:
                mask = mask[..., -1]
            mask = cv2.resize(mask, (512, 512))
        else:
            output = rembg.remove(frame, session=rembg_session)
            mask = np.array(output)[..., 3]

        imageio.imwrite(mask_path, mask)

    # Free raw frames from memory
    del raw_frames

    # --- Stage 2: Mesh reconstruction + hint rendering per frame ---
    print("Stage 2: Mesh reconstruction & hint rendering...")
    hints_dir = os.path.join(cache_dir, "hints")
    os.makedirs(hints_dir, exist_ok=True)

    for i in tqdm(range(total_frames), desc="Mesh recon + hints"):
        frame_hints_dir = os.path.join(hints_dir, f"frame{i:06d}")
        os.makedirs(frame_hints_dir, exist_ok=True)

        # Check if hints already rendered for this frame
        if args.skip_existing and all(
            os.path.exists(os.path.join(frame_hints_dir, f"hint00_{ht}.png"))
            for ht in ["diffuse", "ggx0.05", "ggx0.13", "ggx0.34"]
        ):
            if not use_env_map or os.path.exists(
                os.path.join(frame_hints_dir, "bg00.png")
            ):
                continue

        frame = imageio.v3.imread(
            os.path.join(frames_dir, f"frame{i:06d}.png")
        )
        mask = imageio.v3.imread(
            os.path.join(masks_dir, f"frame{i:06d}.png")
        )
        mask_3ch = mask[..., None].repeat(3, axis=-1) if mask.ndim == 2 else mask

        # Mesh reconstruction
        mesh, fov = mesh_reconstruction(
            frame, mask_3ch, False, shared_fov, args.mask_threshold
        )
        if shared_fov is None:
            shared_fov = float(fov)
            print(f"Auto-estimated FOV from first frame: {shared_fov:.1f}")

        # Get lighting for this frame
        pls = get_light_position(args, i, total_frames)
        env_start_azi = get_env_rotation(args, i, total_frames)

        # Render hints (1 lighting condition → hint00_*.png)
        render_hint_images(
            mesh,
            fov,
            pls,
            args.power,
            env_map=args.env_map,
            env_start_azi=env_start_azi,
            output_folder=frame_hints_dir,
            use_gpu=args.use_gpu_for_rendering,
        )

        if use_env_map:
            render_bg_images(
                fov,
                pls,
                env_map=args.env_map,
                env_start_azi=env_start_azi,
                output_folder=frame_hints_dir,
                use_gpu=args.use_gpu_for_rendering,
            )

        # Clean up temp mesh file
        try:
            os.unlink(mesh)
        except OSError:
            pass

    # --- Stage 3: Relighting generation ---
    # Import here so Blender (stage 2) and diffusion models don't compete for GPU memory
    from demo.relighting_gen import relighting_gen

    print("Stage 3: Relighting generation...")
    for i in tqdm(range(total_frames), desc="Relighting"):
        frame_hints_dir = os.path.join(hints_dir, f"frame{i:06d}")

        # Check if already generated
        if args.skip_existing and os.path.exists(
            os.path.join(frame_hints_dir, "relighting00_0.png")
        ):
            continue

        mask = imageio.v3.imread(os.path.join(masks_dir, f"frame{i:06d}.png"))
        mask_3ch = mask[..., None].repeat(3, axis=-1) if mask.ndim == 2 else mask

        # Use the diffuse hint alpha to mask the input
        mask_for_bg = (
            imageio.v3.imread(
                os.path.join(frame_hints_dir, "hint00_diffuse.png")
            )[..., -1:]
            / 255.0
        )
        frame = imageio.v3.imread(
            os.path.join(frames_dir, f"frame{i:06d}.png")
        )
        masked_image = (
            (frame.astype(np.float32) * mask_for_bg).clip(0, 255).astype(np.uint8)
        )

        relighting_gen(
            masked_ref_img=masked_image,
            mask=mask_3ch,
            cond_path=frame_hints_dir,
            frames=1,
            prompt=args.prompt,
            steps=args.steps,
            seed=args.seed,
            cfg=args.cfg,
            num_imgs_per_prompt=1,
            inpaint=not use_env_map and args.inpaint,
        )

    # --- Stage 4: Assemble output video ---
    print("Stage 4: Assembling output video...")
    all_res = []
    for i in tqdm(range(total_frames), desc="Assembling"):
        frame_hints_dir = os.path.join(hints_dir, f"frame{i:06d}")
        relit_img = imageio.v3.imread(
            os.path.join(frame_hints_dir, "relighting00_0.png")
        )

        if use_env_map:
            mask_for_bg = (
                imageio.v3.imread(
                    os.path.join(frame_hints_dir, "hint00_diffuse.png")
                )[..., -1:]
                / 255.0
            )
            bg = imageio.v3.imread(os.path.join(frame_hints_dir, "bg00.png")) / 255.0
            relit_img = relit_img / 255.0
            relit_img = relit_img * mask_for_bg + bg * (1.0 - mask_for_bg)
            relit_img = (relit_img * 255).clip(0, 255).astype(np.uint8)

        all_res.append(relit_img)

    all_res = np.stack(all_res, axis=0)

    # Get original video FPS
    fps = imageio.v3.immeta(args.input_vid, plugin="pyav").get("fps", 24)

    out_vid = args.out_vid or os.path.splitext(args.input_vid)[0] + "_relit.mp4"
    os.makedirs(os.path.dirname(out_vid) or ".", exist_ok=True)
    imageio.v3.imwrite(out_vid, all_res, fps=fps, codec="libx264", plugin="pyav")
    print(f"Output video saved to {out_vid}")
