import argparse
import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import numpy as np
import torch
from tqdm import tqdm
import librosa as lr
import soundfile as sf

from train_seq import EDGE
from smplx import SMPLX
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
import pickle
import imageio
import cv2
import pyrender
import trimesh
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from moviepy.editor import VideoFileClip, AudioFileClip

def slice_audio(audio_file, stride, length, out_dir):
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx

def extract_features(fpath):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    data, _ = lr.load(fpath, sr=SR)
    envelope = lr.onset.onset_strength(y=data, sr=SR)
    mfcc = lr.feature.mfcc(y=data, sr=SR, n_mfcc=20).T
    chroma = lr.feature.chroma_cens(y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T

    peak_idxs = lr.onset.onset_detect(onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0

    start_bpm = lr.beat.tempo(y=lr.load(fpath)[0])[0]

    tempo, beat_idxs = lr.beat.beat_track(onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH, start_bpm=start_bpm, tightness=100)
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0

    audio_feature = np.concatenate([envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]], axis=-1)
    audio_feature = audio_feature[:4 * FPS]
    return audio_feature

def ensure_directories_exist(directories):
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Creating it.")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"Directory {directory} already exists.")

def generate_motion(opt):
    ensure_directories_exist([opt.motion_save_dir, opt.render_dir])

    feature_func = extract_features
    sample_length = opt.out_length
    stride_ = 60/30
    sample_size = int(sample_length / stride_) - 1
    temp_dir_list = []
    all_cond = []
    all_filenames = []

    print("Computing features for input music")
    songname = os.path.splitext(os.path.basename(opt.music_file))[0]

    temp_dir = TemporaryDirectory()
    temp_dir_list.append(temp_dir)
    dirname = temp_dir.name

    print(f"Slicing {opt.music_file}")
    slice_audio(opt.music_file, 60/30, 120/30, dirname)
    file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1]))

    print(f"Number of slices available: {len(file_list)}")
    if len(file_list) < sample_size:
        sample_size = len(file_list)
        print(f"Adjusting sample size to {sample_size} due to insufficient slices.")

    rand_idx = random.randint(0, len(file_list) - sample_size)
    cond_list = []

    print(f"Computing features for {opt.music_file}")
    for idx, file in enumerate(tqdm(file_list)):
        if not (rand_idx <= idx < rand_idx + sample_size):
            continue
        reps = feature_func(file)[:opt.full_seq_len]
        cond_list.append(reps)
    cond_list = torch.from_numpy(np.array(cond_list))
    all_cond.append(cond_list)
    all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    model = EDGE(opt, opt.feature_type, opt.checkpoint)
    model.eval()

    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir
        Path(fk_out).mkdir(parents=True, exist_ok=True)
        print(f"Generated motion files will be saved in: {fk_out}")

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i]
        print(f"Generating dance for sample {i+1}/{len(all_cond)} with files {all_filenames[i]}")
        model.render_sample(data_tuple, "", opt.render_dir, render_count=-1, fk_out=fk_out, mode="long", render=not opt.no_render)
        print(f"Sample {i+1}/{len(all_cond)} generated and saved.")
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()
    return os.path.join(fk_out, f'_{songname}.pkl')  # Return the full path of the generated motion file

def load_motion_data(file_path):
    """Load motion data from a .npy or .pkl file."""
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            smpl_poses = data["smpl_poses"]
            modata = np.concatenate((data["smpl_trans"], smpl_poses), axis=1)
            if modata.shape[1] == 69:
                hand_zeros = np.zeros([modata.shape[0], 90], dtype=np.float32)
                modata = np.concatenate((modata, hand_zeros), axis=1)
            assert modata.shape[1] == 159
            modata[:, 1] = modata[:, 1] + 1.3
            return modata
    else:
        raise ValueError("Unsupported file format. Only .npy and .pkl are supported.")

def rot6d_to_axis_angle(rot6d):
    """Convert 6D rotation to axis-angle representation."""
    rot6d = rot6d.clone().detach().float()  # Updated to avoid the warning
    batch_size = rot6d.shape[0]
    num_joints = rot6d.shape[1] // 6
    rot6d = rot6d.view(batch_size, num_joints, 6)
    rotation_matrices = rotation_6d_to_matrix(rot6d)
    axis_angles = matrix_to_axis_angle(rotation_matrices.view(-1, 3, 3)).view(batch_size, num_joints, 3)
    return axis_angles.numpy().reshape(batch_size, -1)

class MotionVisualizer():
    def __init__(self, save_path, smplx_path, img_size=(500, 500), fps=30, gpu='0'):
        self.save_path = save_path
        self.img_size = img_size
        self.fps = fps
        
        # Check if GPU index is valid
        if not torch.cuda.is_available() or int(gpu) >= torch.cuda.device_count():
            raise ValueError(f"Invalid GPU index: {gpu}. Available GPUs: {torch.cuda.device_count()}")
        
        self.smplx = SMPLX(smplx_path, use_pca=False, flat_hand_mean=True).eval().to(f'cuda:{gpu}')

        self.scene = pyrender.Scene()
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [0, 0, 4]  # Adjust camera position
        self.scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.scene.add(light, pose=camera_pose)
        self.renderer = pyrender.OffscreenRenderer(self.img_size[0], self.img_size[1])

    def save_video(self, save_path, color_list):
        f = cv2.VideoWriter_fourcc(*'mp4v')
        videowriter = cv2.VideoWriter(save_path, f, self.fps, self.img_size)
        for frame in color_list:
            videowriter.write(frame[:, :, ::-1])
        videowriter.release()

    def render_frames(self, meshes, batch_size=40):
        color_list = []
        for i in tqdm(range(0, len(meshes), batch_size)):
            batch_meshes = meshes[i:i+batch_size]
            for mesh in batch_meshes:
                mesh_node = self.scene.add(pyrender.Mesh.from_trimesh(mesh))
                color, _ = self.renderer.render(self.scene)
                color_list.append(color)
                self.scene.remove_node(mesh_node)
                torch.cuda.empty_cache()  # Free up GPU memory
        return color_list

    def visualize(self, motion_data):
        output = self.smplx.forward(
            betas=torch.zeros([motion_data.shape[0], self.smplx.num_betas]).to(motion_data.device),
            transl=motion_data[:, :3],
            global_orient=motion_data[:, 3:6],
            body_pose=motion_data[:, 6:69],
            jaw_pose=torch.zeros([motion_data.shape[0], 3]).to(motion_data),
            leye_pose=torch.zeros([motion_data.shape[0], 3]).to(motion_data),
            reye_pose=torch.zeros([motion_data.shape[0], 3]).to(motion_data),
            left_hand_pose=motion_data[:, 69:114],
            right_hand_pose=motion_data[:, 114:],
            expression=torch.zeros([motion_data.shape[0], 10]).to(motion_data)
        )

        meshes = [trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces) for i in range(output.vertices.shape[0])]
        color_list = self.render_frames(meshes)
        video_path = os.path.join(self.save_path, 'motion_visualization.mp4')
        self.save_video(video_path, color_list)
        return video_path

    def add_music_to_video(self, video_path, music_path):
        print(video_path)
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(music_path).subclip(0, 120)
        final_clip = video_clip.set_audio(audio_clip)
        music_name = os.path.splitext(os.path.basename(music_path))[0]
        output_video_path = os.path.join(self.save_path, f'motion_visualization_with_{music_name}.mp4')
        final_clip.write_videofile(output_video_path)

def motion_data_load_process(motionfile):
    modata = load_motion_data(motionfile)
    if modata.shape[-1] == 315:
        rot6d = torch.from_numpy(modata[:, 3:])
        T, C = rot6d.shape
        rot6d = rot6d.reshape(-1, 6)
        axis = rot6d_to_axis_angle(rot6d).reshape(T, -1)
        modata = np.concatenate((modata[:, :3], axis), axis=1)
    return modata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and Visualize Motion from Music")
    parser.add_argument("--music_file", type=str, required=True, help="Path to the input music file")
    parser.add_argument("--motion_save_dir", type=str, required=True, help="Directory to save motion files")
    parser.add_argument("--render_dir", type=str, required=True, help="Directory to save rendered files")
    parser.add_argument("--out_length", type=int, default=120, help="Output length in frames")
    parser.add_argument("--full_seq_len", type=int, default=120, help="Full sequence length for feature extraction")
    parser.add_argument("--feature_type", type=str, default="mfcc", help="Type of features to use")
    parser.add_argument("--checkpoint", type=str, default="assets/checkpoints/train-2000.pt", help="Path to the model checkpoint")
    parser.add_argument("--nfeats", type=int, default=319, help="Number of features")
    parser.add_argument("--save_motions", action="store_true", help="Save generated motions")
    parser.add_argument("--no_render", action="store_true", help="Do not render the video")
    parser.add_argument("--gpu", type=str, default="0", help="GPU index to use")
    parser.add_argument("--smplx_path", type=str, required=True, help="Path to the SMPLX model")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video")
    parser.add_argument("--visualize_save_dir", type=str, required=True, help="Directory to save the visualized motion video")
    parser.add_argument(
        "--do_normalize",
        action="store_true",
        help="normalize")

    opt = parser.parse_args()
    
    # Generate motion and get the path of the generated motion file
    motion_file_path = generate_motion(opt)
    print(motion_file_path)
    
    # Visualize motion
    visualizer = MotionVisualizer(save_path=opt.visualize_save_dir, smplx_path=opt.smplx_path, fps=opt.fps, gpu=opt.gpu)
    motion_data = motion_data_load_process(motion_file_path)
    motion_data = torch.tensor(motion_data, dtype=torch.float32, device=f'cuda:{opt.gpu}')
    video_path = visualizer.visualize(motion_data)
    visualizer.add_music_to_video(video_path,opt.music_file)
