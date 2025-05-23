# AI Choreographer
AI Choreographer is an AI-powered dance generation system that uses a GAN architecture with LSTM layers to learn and generate motion sequences that align with the beat and rhythm of music. It leverages the SMPL-X body model to visualize the generated motion as realistic 3D human animation, ultimately exporting the output as a video.

This project explores the intersection of music understanding, generative models, and human motion synthesis, aiming to automate choreography with rhythm-aware movement generation.

## Key Features
🎶 Beat-synchronized dance generation from raw music input

🧠 GAN architecture enhanced with LSTM for sequential motion learning

🕴️ Realistic human visualization using the SMPL-X model

🎥 Final output saved as a video of the generated dance


## REPOSITORY STRUCTURE
    
    ├── finedance_dataset.py       # Preprocess and synchronize motion/music files
    ├── args.py                    # Defines all command-line arguments
    ├── gancontinueresume.py       # GAN model with LSTM architecture for motion generation
    ├── combined.py                # Integrates the system: takes music, generates motion, visualizes and saves video


## Model Overview
Generator & Discriminator: Trained to generate plausible dance sequences from music features.

LSTM Layer: Captures the temporal dynamics of dance aligned with music beats.

SMPL-X Model: Used to render generated motion as a 3D humanoid and save animations.

## How to Use
Prepare your dataset of paired music and motion files.

Preprocess the data using:

    python finedance_dataset.py
Train or resume the GAN model using:

    python gancontinueresume.py --your-args-here
Generate dance from user input music and visualize it:

    python combined.py --music_path your_music_file.wav
The final motion sequence will be visualized using SMPL-X and saved as a video file. 

## Team Members
VAISAKH

MUSTHAFA

FASEEH ALI

SHAZZABRIN 

## Attribution and License Notice
This project is based on the FineDance repository by Ronghui Li et al., available at:
https://github.com/li-ronghui/FineDance

The original code is licensed under the MIT License. Please refer to the original repository for full license details.


##  Citation

This project uses the FineDance dataset for training and evaluation. If you use this dataset, please cite the following paper:

```bibtex
@InProceedings{Li_2023_ICCV,
    author    = {Li, Ronghui and Zhao, Junfan and Zhang, Yachao and Su, Mingyang and Ren, Zeping and Zhang, Han and Tang, Yansong and Li, Xiu},
    title     = {FineDance: A Fine-grained Choreography Dataset for 3D Full Body Dance Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {10234-10243}
}


