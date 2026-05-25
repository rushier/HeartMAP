import torch
import torchvision


class NetGenerator:

    def __init__(self, args):
        self.args = args

    def EchoPrime(self, checkpoint='./models/pretrained/echo_prime_encoder.pt'):
        print('using EchoPrime')
        echo_encoder = torchvision.models.video.mvit_v2_s()
        echo_encoder.head[-1] = torch.nn.Linear(echo_encoder.head[-1].in_features, 512)
        if checkpoint is not None:
            pretrain = torch.load(checkpoint, weights_only=False)
            echo_encoder.load_state_dict(pretrain)
            print('loaded model weights:', checkpoint)
        return echo_encoder
