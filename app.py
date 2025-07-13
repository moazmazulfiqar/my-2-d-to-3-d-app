import gradio as gr
import torch
from PIL import Image
import numpy as np
from diffusers import AnimateDiffControlNetPipeline, ControlNetModel, MotionAdapter, AutoencoderKL, LCMScheduler
from controlnet_aux import CannyDetector, HEDdetector
import ffmpeg

def load_pipeline():
    controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    motion_adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    pipe = AnimateDiffControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet_canny,
        motion_adapter=motion_adapter,
        vae=vae,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    return pipe, controlnet_canny, controlnet_depth

pipe, cn_canny, cn_depth = load_pipeline()
canny_det = CannyDetector()
hed_det = HEDdetector()

def generate(
    image: Image.Image,
    mode: str,
    prompt: str,
    num_frames: int = 16,
    fps: int = 8,
):
    img = image.convert("RGB")
    img_arr = np.array(img)
    if mode == "Canny":
        cond = canny_det.detect(img_arr)
        pipe.controlnet = cn_canny
    else:
        cond = hed_det.detect(img_arr)
        pipe.controlnet = cn_depth
    
    out = pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry",
        num_frames=num_frames,
        num_inference_steps=25,
        controlnet_conditioning_image=cond,
        controlnet_conditioning_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    frames = out.frames[0]  # PIL Images
    
    # Save MP4
    video_path = "/mnt/data/animation.mp4"
    frames[0].save(
        video_path, save_all=True, append_images=frames[1:], fps=fps, format="mp4"
    )
    return video_path

title = "🖼️ 2D-to-3D AI Animation"
description = "Upload your sketch, choose guidance, add a prompt, and generate a short animated MP4."

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(type="pil", label="Sketch or Image"),
        gr.Radio(["Canny", "Depth"], label="Control Mode"),
        gr.Textbox(lines=1, placeholder="e.g. anime girl dancing", label="Prompt"),
        gr.Slider(8, 32, 16, label="Num Frames"),
        gr.Slider(4, 30, 8, label="FPS"),
    ],
    outputs=gr.Video(label="Generated Animation"),
    title=title,
    description=description,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
