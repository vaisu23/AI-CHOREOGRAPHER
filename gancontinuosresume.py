import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.FineDance_dataset import FineDance_Smpl
from args import FineDance_parse_train_opt
from tqdm import tqdm
import os
import re
import librosa

class Generator(nn.Module):
    def __init__(self, music_dim, condition_dim, hidden_dim=256, output_dim=315, num_layers=3):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(music_dim + condition_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x, condition):
        batch_size, seq_len, feature_dim = x.size()
        condition_expanded = condition.unsqueeze(1).expand(batch_size, seq_len, condition.size(1))
        x = torch.cat((x, condition_expanded), dim=2)
        output, _ = self.lstm(x)
        output = output.contiguous().view(-1, output.size(2))
        output = self.fc(output)
        output = output.view(batch_size, seq_len, -1)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_dim=315, condition_dim=10, hidden_dim=256, num_layers=3):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim + condition_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, condition):
        batch_size, seq_len, feature_dim = x.size()
        condition_expanded = condition.unsqueeze(1).expand(batch_size, seq_len, condition.size(1))
        x = torch.cat((x, condition_expanded), dim=2)
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        return self.fc(output)

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return onset_env, chroma, mfcc

def find_latest_checkpoint(directory="."):
    checkpoint_files = [f for f in os.listdir(directory) if re.match(r'checkpoint_epoch_\d+\.pth', f)]
    if not checkpoint_files:
        return None, 0
    checkpoint_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    latest_checkpoint = checkpoint_files[-1]
    start_epoch = int(re.search(r'\d+', latest_checkpoint).group())
    return latest_checkpoint, start_epoch

def resume_training(train_data_loader, music_dim, motion_dim, condition_dim, additional_epochs=100, lr_g=0.00005, lr_d=0.00002):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator(music_dim=music_dim, condition_dim=condition_dim, hidden_dim=256, output_dim=315, num_layers=3).to(device)
    discriminator = Discriminator(input_dim=motion_dim, condition_dim=condition_dim, hidden_dim=256, num_layers=3).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d)

    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()

    latest_checkpoint, start_epoch = find_latest_checkpoint()
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, weights_only=True)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        print(f"Checkpoint loaded: {latest_checkpoint}")

    end_epoch = start_epoch + additional_epochs

    for epoch in range(start_epoch, end_epoch):
        g_loss_epoch = 0
        d_loss_epoch = 0
        with tqdm(total=len(train_data_loader), desc=f"Epoch {epoch+1}/{end_epoch}") as pbar: 
            for i, (motion, music, condition, filename) in enumerate(train_data_loader):
                batch_size = music.size(0)

                motion = motion.view(batch_size, 120, -1).to(device)
                music = music.view(batch_size, 120, -1).to(device)
                condition = condition.to(device)

                valid = torch.ones(batch_size, 1).to(device)
                fake = torch.zeros(batch_size, 1).to(device)

                optimizer_G.zero_grad()
                z = music
                generated_motion = generator(z, condition)
                
                g_loss = criterion(discriminator(generated_motion, condition), valid)
                g_loss.backward()
                optimizer_G.step()

                optimizer_D.zero_grad()
                real_loss = criterion(discriminator(motion, condition), valid)
                fake_loss = criterion(discriminator(generated_motion.detach(), condition), fake)
                
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()
                pbar.update(1)

        scheduler_G.step()
        scheduler_D.step()

        if g_loss_epoch > d_loss_epoch * 5:
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = lr_d * 5
        else:
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = lr_d

        assert generated_motion.shape == motion.shape, f"Generated motion shape {generated_motion.shape} does not match motion shape {motion.shape}"
        assert discriminator(generated_motion, condition).shape == valid.shape, f"Discriminator output shape {discriminator(generated_motion, condition).shape} does not match valid shape {valid.shape}"
        assert discriminator(motion, condition).shape == valid.shape, f"Discriminator output shape {discriminator(motion, condition).shape} does not match valid shape {valid.shape}"
        assert discriminator(generated_motion.detach(), condition).shape == fake.shape, f"Discriminator output shape {discriminator(generated_motion.detach(), condition).shape} does not match fake shape {fake.shape}"

        print(f"Epoch {epoch+1}/{end_epoch}: D_loss={d_loss_epoch/len(train_data_loader)}, G_loss={g_loss_epoch/len(train_data_loader)}")

        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'g_loss': g_loss_epoch / len(train_data_loader),
                'd_loss': d_loss_epoch / len(train_data_loader)
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")

    return generator, discriminator

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    opt = FineDance_parse_train_opt()

    train_dataset = FineDance_Smpl(args=opt, istrain=True)
    train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    music_dim = 35
    motion_dim = 315
    condition_dim = 10 

    additional_epochs = 100

    generator, discriminator = resume_training(train_data_loader, music_dim, motion_dim, condition_dim, additional_epochs=additional_epochs)
    torch.save(generator.state_dict(), "resumed_generator200.pth")
    print("Resumed generator model saved as resumed_generator.pth")