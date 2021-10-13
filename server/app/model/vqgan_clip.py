# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun
# Adapted from https://github.com/nerdyrodent/VQGAN-CLIP/blob/main/generate.py

import argparse
import os
import random
from urllib.request import urlopen

import imageio
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
torch.backends.cudnn.benchmark = False		# NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)	# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True

from subprocess import Popen, PIPE
import re

from model.utils import *


IMAGE_SIZE = 128

class Args(BaseModel):
    prompts: str = None
    image_prompts: list = []
    size: list = [IMAGE_SIZE, IMAGE_SIZE]
    init_image: str = None
    init_noise: str = None
    init_weight: float = 0.
    clip_model: str = "ViT-B/32"
    vqgan_config: str = "model/checkpoints/vqgan_imagenet_f16_16384.yaml"
    vqgan_checkpoint: str = "model/checkpoints/vqgan_imagenet_f16_16384.yaml"
    noise_prompt_seeds: list = []
    noise_prompt_weights: list = []
    step_size: float = 0.1
    cut_method: str = "latest"
    cutn: int = 32
    cut_pow: float = 1.
    seed: int = None
    optimiser: str = "Adam"
    output: str = "output.png"
    make_video: bool = False
    make_zoom_video: bool = False
    zoom_start: int = 0
    zoom_frequency: int = 10
    zoom_scale: float = 0.99
    zoom_shift_x: int = 0
    zoom_shift_y: int = 0
    prompt_frequency: int = 0
    video_length: float = 10.
    output_video_fps: float = 0.
    input_video_fps: float = 15.
    cudnn_determinism: bool = False
    augments: list = []
    video_style_dir: str = None
    cuda_device: str = "cuda:0"



def load_model(cuda_device=0, vqgan_config="model/checkpoints/vqgan_imagenet_f16_16384.yaml", vqgan_checkpoint="model/checkpoints/vqgan_imagenet_f16_16384.ckpt"):
    device = torch.device(cuda_device)
    return load_vqgan_model(vqgan_config, vqgan_checkpoint, gumbel=False).to(device)


def load_perceptor(cuda_device=0, clip_model="ViT-B/32"):
    device = torch.device(cuda_device)
    jit = True if float(torch.__version__[:3]) < 1.8 else False
    return clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)


def generate(
    model,
    perceptor,
    output_path="output.png",
    cuda_device="cuda:0",
    prompts=None,
    iterations=500, 
    save_every=500, 
    size=[IMAGE_SIZE, IMAGE_SIZE],
    gumbel=False,
    ):

    # Create the parser
    vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

    # Add the arguments
    vq_parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", default=None, dest='prompts')
    vq_parser.add_argument("-ip",   "--image_prompts", type=str, help="Image prompts / target image", default=[], dest='image_prompts')
    vq_parser.add_argument("-s",    "--size", nargs=2, type=int, help="Image size (width height) (default: %(default)s)", default=[IMAGE_SIZE,IMAGE_SIZE], dest='size')
    vq_parser.add_argument("-ii",   "--init_image", type=str, help="Initial image", default=None, dest='init_image')
    vq_parser.add_argument("-in",   "--init_noise", type=str, help="Initial noise image (pixels or gradient)", default=None, dest='init_noise')
    vq_parser.add_argument("-iw",   "--init_weight", type=float, help="Initial weight", default=0., dest='init_weight')
    vq_parser.add_argument("-m",    "--clip_model", type=str, help="CLIP model (e.g. ViT-B/32, ViT-B/16)", default='ViT-B/32', dest='clip_model')
    vq_parser.add_argument("-conf", "--vqgan_config", type=str, help="VQGAN config", default=f'checkpoints/vqgan_imagenet_f16_16384.yaml', dest='vqgan_config')
    vq_parser.add_argument("-ckpt", "--vqgan_checkpoint", type=str, help="VQGAN checkpoint", default=f'checkpoints/vqgan_imagenet_f16_16384.ckpt', dest='vqgan_checkpoint')
    vq_parser.add_argument("-nps",  "--noise_prompt_seeds", nargs="*", type=int, help="Noise prompt seeds", default=[], dest='noise_prompt_seeds')
    vq_parser.add_argument("-npw",  "--noise_prompt_weights", nargs="*", type=float, help="Noise prompt weights", default=[], dest='noise_prompt_weights')
    vq_parser.add_argument("-lr",   "--learning_rate", type=float, help="Learning rate", default=0.1, dest='step_size')
    vq_parser.add_argument("-cutm", "--cut_method", type=str, help="Cut method", choices=['original','updated','nrupdated','updatedpooling','latest'], default='latest', dest='cut_method')
    vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest='cutn')
    vq_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')
    vq_parser.add_argument("-sd",   "--seed", type=int, help="Seed", default=None, dest='seed')
    vq_parser.add_argument("-opt",  "--optimiser", type=str, help="Optimiser", choices=['Adam','AdamW','Adagrad','Adamax','DiffGrad','AdamP','RAdam','RMSprop'], default='Adam', dest='optimiser')
    vq_parser.add_argument("-o",    "--output", type=str, help="Output filename", default="output.png", dest='output')
    vq_parser.add_argument("-vid",  "--video", action='store_true', help="Create video frames?", dest='make_video')
    vq_parser.add_argument("-zvid", "--zoom_video", action='store_true', help="Create zoom video?", dest='make_zoom_video')
    vq_parser.add_argument("-zs",   "--zoom_start", type=int, help="Zoom start iteration", default=0, dest='zoom_start')
    vq_parser.add_argument("-zse",  "--zoom_save_every", type=int, help="Save zoom image iterations", default=10, dest='zoom_frequency')
    vq_parser.add_argument("-zsc",  "--zoom_scale", type=float, help="Zoom scale %", default=0.99, dest='zoom_scale')
    vq_parser.add_argument("-zsx",  "--zoom_shift_x", type=int, help="Zoom shift x (left/right) amount in pixels", default=0, dest='zoom_shift_x')
    vq_parser.add_argument("-zsy",  "--zoom_shift_y", type=int, help="Zoom shift y (up/down) amount in pixels", default=0, dest='zoom_shift_y')
    vq_parser.add_argument("-cpe",  "--change_prompt_every", type=int, help="Prompt change frequency", default=0, dest='prompt_frequency')
    vq_parser.add_argument("-vl",   "--video_length", type=float, help="Video length in seconds (not interpolated)", default=10, dest='video_length')
    vq_parser.add_argument("-ofps", "--output_video_fps", type=float, help="Create an interpolated video (Nvidia GPU only) with this fps (min 10. best set to 30 or 60)", default=0, dest='output_video_fps')
    vq_parser.add_argument("-ifps", "--input_video_fps", type=float, help="When creating an interpolated video, use this as the input fps to interpolate from (>0 & <ofps)", default=15, dest='input_video_fps')
    vq_parser.add_argument("-d",    "--deterministic", action='store_true', help="Enable cudnn.deterministic?", dest='cudnn_determinism')
    vq_parser.add_argument("-aug",  "--augments", nargs='+', action='append', type=str, choices=['Ji','Sh','Gn','Pe','Ro','Af','Et','Ts','Cr','Er','Re'], help="Enabled augments (latest vut method only)", default=[], dest='augments')
    vq_parser.add_argument("-vsd",  "--video_style_dir", type=str, help="Directory with video frames to style", default=None, dest='video_style_dir')
    vq_parser.add_argument("-cd",   "--cuda_device", type=str, help="Cuda device to use", default="cuda:0", dest='cuda_device')


    # Execute the parse_args() method
    # args = vq_parser.parse_args()
    args = Args()

    if not prompts and not args.image_prompts:
        raise Exception("No prompt received")

    if args.cudnn_determinism:
        torch.backends.cudnn.deterministic = True

    if not args.augments:
        args.augments = [['Af', 'Pe', 'Ji', 'Er']]

    # Split text prompts using the pipe character (weights are split later)
    if prompts:
        # For stories, there will be many phrases
        story_phrases = [phrase.strip() for phrase in prompts.split("^")]
        
        # Make a list of all phrases
        all_phrases = []
        for phrase in story_phrases:
            all_phrases.append(phrase.split("|"))
        
        # First phrase
        prompts = all_phrases[0]
        
    # Split target images using the pipe character (weights are split later)
    if args.image_prompts:
        args.image_prompts = args.image_prompts.split("|")
        args.image_prompts = [image.strip() for image in args.image_prompts]

    if args.make_video and args.make_zoom_video:
        print("Warning: Make video and make zoom video are mutually exclusive.")
        args.make_video = False
        
    # Make video steps directory
    if args.make_video or args.make_zoom_video:
        if not os.path.exists('steps'):
            os.mkdir('steps')

    # Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
    # NB. May not work for AMD cards?
    if not args.cuda_device == 'cpu' and not torch.cuda.is_available():
        args.cuda_device = 'cpu'
        args.video_fps = 0
        print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
        print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")

    # If a video_style_dir has been, then create a list of all the images
    if args.video_style_dir:
        print("Locating video frames...")
        video_frame_list = []
        for entry in os.scandir(args.video_style_dir):
            if (entry.path.endswith(".jpg")
                    or entry.path.endswith(".png")) and entry.is_file():
                video_frame_list.append(entry.path)

        # Reset a few options - same filename, different directory
        if not os.path.exists('steps'):
            os.mkdir('steps')

        args.init_image = video_frame_list[0]
        filename = os.path.basename(args.init_image)
        cwd = os.getcwd()
        output_path = os.path.join(cwd, "steps", filename)
        num_video_frames = len(video_frame_list) # for video styling

    device = torch.device(cuda_device)

    # clock=deepcopy(perceptor.visual.positional_embedding.data)
    # perceptor.visual.positional_embedding.data = clock/clock.max()
    # perceptor.visual.positional_embedding.data=clamp_with_grad(clock,0,1)

    cut_size = perceptor.visual.input_resolution
    f = 2**(model.decoder.num_resolutions - 1)

    # Cutout class options:
    # 'latest','original','updated' or 'updatedpooling'
    if args.cut_method == 'latest':
        make_cutouts = MakeCutouts(args, cut_size, args.cutn, cut_pow=args.cut_pow)
    elif args.cut_method == 'original':
        make_cutouts = MakeCutoutsOrig(args, cut_size, args.cutn, cut_pow=args.cut_pow)
    elif args.cut_method == 'updated':
        make_cutouts = MakeCutoutsUpdate(args, cut_size, args.cutn, cut_pow=args.cut_pow)
    elif args.cut_method == 'nrupdated':
        make_cutouts = MakeCutoutsNRUpdate(args, cut_size, args.cutn, cut_pow=args.cut_pow)
    else:
        make_cutouts = MakeCutoutsPoolingUpdate(args, cut_size, args.cutn, cut_pow=args.cut_pow)    

    toksX, toksY = size[0] // f, size[1] // f
    sideX, sideY = toksX * f, toksY * f

    # Gumbel or not?
    if gumbel:
        e_dim = 256
        n_toks = model.quantize.n_embed
        z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
    else:
        e_dim = model.quantize.e_dim
        n_toks = model.quantize.n_e
        z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


    if args.init_image:
        if 'http' in args.init_image:
            img = Image.open(urlopen(args.init_image))
        else:
            img = Image.open(args.init_image)
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    elif args.init_noise == 'pixels':
        img = random_noise_image(size[0], size[1])    
        pil_image = img.convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    elif args.init_noise == 'gradient':
        img = random_gradient_image(size[0], size[1])
        pil_image = img.convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        # z = one_hot @ model.quantize.embedding.weight
        if gumbel:
            z = one_hot @ model.quantize.embed.weight
        else:
            z = one_hot @ model.quantize.embedding.weight

        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2) 
        #z = torch.rand_like(z)*2						# NR: check

    z_orig = z.clone()
    z.requires_grad_(True)

    pMs = []
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    # From imagenet - Which is better?
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # CLIP tokenize/encode   
    if prompts:
        for prompt in prompts:
            txt, weight, stop = split_prompt(prompt)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = split_prompt(prompt)
        img = Image.open(path)
        pil_image = img.convert('RGB')
        img = resize_image(pil_image, (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))


    # Set the optimiser
    opt = get_opt(args.optimiser, args.step_size, z)


    # Output for the user
    print('Using device:', device)
    print('Optimising using:', args.optimiser)

    if prompts:
        print('Using text prompts:', prompts)  
    if args.image_prompts:
        print('Using image prompts:', args.image_prompts)
    if args.init_image:
        print('Using initial image:', args.init_image)
    if args.noise_prompt_weights:
        print('Noise prompt weights:', args.noise_prompt_weights)    


    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed  
    torch.manual_seed(seed)
    print('Using seed:', seed)

    i = 0 # Iteration counter
    j = 0 # Zoom video frame counter
    p = 1 # Phrase counter
    smoother = 0 # Smoother counter
    this_video_frame = 0 # for video styling

    # Messing with learning rate / optimisers
    #variable_lr = args.step_size
    #optimiser_list = [['Adam',0.075],['AdamW',0.125],['Adagrad',0.2],['Adamax',0.125],['DiffGrad',0.075],['RAdam',0.125],['RMSprop',0.02]]

    # Do it
    try:
        with tqdm() as pbar:
            while True:            
                # Change generated image
                if args.make_zoom_video:
                    if i % args.zoom_frequency == 0:
                        out = synth(z)
                        
                        # Save image
                        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
                        img = np.transpose(img, (1, 2, 0))
                        imageio.imwrite('./steps/' + str(j) + '.png', np.array(img))

                        # Time to start zooming?                    
                        if args.zoom_start <= i:
                            # Convert z back into a Pil image                    
                            #pil_image = TF.to_pil_image(out[0].cpu())
                            
                            # Convert NP to Pil image
                            pil_image = Image.fromarray(np.array(img).astype('uint8'), 'RGB')
                                                    
                            # Zoom
                            if args.zoom_scale != 1:
                                pil_image_zoom = zoom_at(pil_image, sideX/2, sideY/2, args.zoom_scale)
                            else:
                                pil_image_zoom = pil_image
                            
                            # Shift - https://pillow.readthedocs.io/en/latest/reference/ImageChops.html
                            if args.zoom_shift_x or args.zoom_shift_y:
                                # This one wraps the image
                                pil_image_zoom = ImageChops.offset(pil_image_zoom, args.zoom_shift_x, args.zoom_shift_y)
                            
                            # Convert image back to a tensor again
                            pil_tensor = TF.to_tensor(pil_image_zoom)
                            
                            # Re-encode
                            z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                            z_orig = z.clone()
                            z.requires_grad_(True)

                            # Re-create optimiser
                            opt = get_opt(args.optimiser, args.step_size)
                        
                        # Next
                        j += 1
                
                # Change text prompt
                if args.prompt_frequency > 0:
                    if i % args.prompt_frequency == 0 and i > 0:
                        # In case there aren't enough phrases, just loop
                        if p >= len(all_phrases):
                            p = 0
                        
                        pMs = []
                        prompts = all_phrases[p]

                        # Show user we're changing prompt                                
                        print(prompts)
                        
                        for prompt in prompts:
                            txt, weight, stop = split_prompt(prompt)
                            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                            pMs.append(Prompt(embed, weight, stop).to(device))
                                            
                        '''
                        # Smooth test
                        smoother = args.zoom_frequency * 15 # smoothing over x frames
                        variable_lr = args.step_size * 0.25
                        opt = get_opt(args.optimiser, variable_lr)
                        '''
                        
                        p += 1
                
                '''
                if smoother > 0:
                    if smoother == 1:
                        opt = get_opt(args.optimiser, args.step_size)
                    smoother -= 1
                '''
                
                '''
                # Messing with learning rate / optimisers
                if i % 225 == 0 and i > 0:
                    variable_optimiser_item = random.choice(optimiser_list)
                    variable_optimiser = variable_optimiser_item[0]
                    variable_lr = variable_optimiser_item[1]
                    
                    opt = get_opt(variable_optimiser, variable_lr)
                    print("New opt: %s, lr= %f" %(variable_optimiser,variable_lr)) 
                '''
                

                # Training time
                train(pMs, opt, i, z, model, perceptor, make_cutouts, normalize, z_min, z_max, save_every, args.init_weight, args.make_video, prompts, output_path, gumbel)
                
                # Ready to stop yet?
                if i == iterations:
                    if not args.video_style_dir:
                        # we're done
                        break
                    else:                    
                        if this_video_frame == (num_video_frames - 1):
                            # we're done
                            make_styled_video = True
                            break
                        else:
                            # Next video frame
                            this_video_frame += 1

                            # Reset the iteration count
                            i = -1
                            pbar.reset()
                                                    
                            # Load the next frame, reset a few options - same filename, different directory
                            args.init_image = video_frame_list[this_video_frame]
                            print("Next frame: ", args.init_image)

                            if args.seed is None:
                                seed = torch.seed()
                            else:
                                seed = args.seed  
                            torch.manual_seed(seed)
                            print("Seed: ", seed)

                            filename = os.path.basename(args.init_image)
                            output_path = os.path.join(cwd, "steps", filename)

                            # Load and resize image
                            img = Image.open(args.init_image)
                            pil_image = img.convert('RGB')
                            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
                            pil_tensor = TF.to_tensor(pil_image)
                            
                            # Re-encode
                            z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                            z_orig = z.clone()
                            z.requires_grad_(True)

                            # Re-create optimiser
                            opt = get_opt(args.optimiser, args.step_size)

                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass

    # All done :)

    # Video generation
    if args.make_video or args.make_zoom_video:
        init_frame = 1      # Initial video frame
        if args.make_zoom_video:
            last_frame = j
        else:
            last_frame = i  # This will raise an error if that number of frames does not exist.

        length = args.video_length # Desired time of the video in seconds

        min_fps = 10
        max_fps = 60

        total_frames = last_frame-init_frame

        frames = []
        tqdm.write('Generating video...')
        for i in range(init_frame,last_frame):
            temp = Image.open("./steps/"+ str(i) +'.png')
            keep = temp.copy()
            frames.append(keep)
            temp.close()
        
        if args.output_video_fps > 9:
            # Hardware encoding and video frame interpolation
            print("Creating interpolated frames...")
            ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps={args.output_video_fps}'"
            output_file = re.compile('\.png$').sub('.mp4', output_path)
            try:
                p = Popen(['ffmpeg',
                        '-y',
                        '-f', 'image2pipe',
                        '-vcodec', 'png',
                        '-r', str(args.input_video_fps),               
                        '-i',
                        '-',
                        '-b:v', '10M',
                        '-vcodec', 'h264_nvenc',
                        '-pix_fmt', 'yuv420p',
                        '-strict', '-2',
                        '-filter:v', f'{ffmpeg_filter}',
                        '-metadata', f'comment={prompts}',
                    output_file], stdin=PIPE)
            except FileNotFoundError:
                print("ffmpeg command failed - check your installation")
            for im in tqdm(frames):
                im.save(p.stdin, 'PNG')
            p.stdin.close()
            p.wait()
        else:
            # CPU
            fps = np.clip(total_frames/length,min_fps,max_fps)
            output_file = re.compile('\.png$').sub('.mp4', output_path)
            try:
                p = Popen(['ffmpeg',
                        '-y',
                        '-f', 'image2pipe',
                        '-vcodec', 'png',
                        '-r', str(fps),
                        '-i',
                        '-',
                        '-vcodec', 'libx264',
                        '-r', str(fps),
                        '-pix_fmt', 'yuv420p',
                        '-crf', '17',
                        '-preset', 'veryslow',
                        '-metadata', f'comment={prompts}',
                        output_file], stdin=PIPE)
            except FileNotFoundError:
                print("ffmpeg command failed - check your installation")        
            for im in tqdm(frames):
                im.save(p.stdin, 'PNG')
            p.stdin.close()
            p.wait()     
