from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
from torchvision.io import read_video
from Resnet101 import ResNet101




'''
VisExtNet is made up of a MTCNN and a VGGFace2-pretrained ResNet101. Firstly, MTCNN identifies the faces of interlocutors
appeared in each frame of the video, then a ResNet101 pretrained on VGGFace2, a large-scale face dataset, is utilized to extract 
facial visual features of each human face. Finally, the output features from each frame are average pooled over the frame axis
to obtain the final visual representation of that video.
'''
class VisExtNet(nn.Module):

    def __init__(self, resnet_weight_path, feature_dim, device):
        super().__init__()

        self.mtcnn = MTCNN(image_size = 224, keep_all = True, device = device)
        self.resnet = ResNet101().to(device)
        self.resnet.load_state_dict(torch.load(resnet_weight_path))
        self.feature_dim = feature_dim
        self.device = device
    

    def forward(self, video_path):
        video_frames, _, _ = read_video(filename = video_path)
        video_faces = self.mtcnn(video_frames)
        video_faces_delete_none_scene = [video_face for video_face in video_faces if video_face is not None]
        num_frames = len(video_faces_delete_none_scene)
        if num_frames == 0:
            return torch.zeros(self.feature_dim).to(self.device)
        elif num_frames <= 20:
            all_video_faces = torch.cat(video_faces_delete_none_scene, dim = 0).to(self.device)
        else:
            steps = int(num_frames / 20)
            all_video_faces = torch.cat(video_faces_delete_none_scene[::steps][:20], dim = 0).to(self.device)
        visual_embedding = torch.mean(self.resnet(all_video_faces), dim = 0)

        return visual_embedding
    



