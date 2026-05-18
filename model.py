from typing import List

from encoder import SEANetEncoder
from decoder import VocosBackbone, ISTFTHead
from vq import ResidualVectorQuantizer
import torch    
import torch.nn as nn

# Change the arguments keep structure similar

class HOA_WavTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SEANetEncoder(causal=False, 
                                    n_residual_layers=1, 
                                    norm='weight_norm', 
                                    pad_mode='reflect', 
                                    lstm=2,
                                    dimension=512, 
                                    channels=1, 
                                    n_filters=32, 
                                    ratios=[8, 5, 4, 2], 
                                    activation='ELU',
                                    kernel_size=7, 
                                    residual_kernel_size=3, 
                                    last_kernel_size=7, 
                                    dilation_base=2,
                                    true_skip=False, 
                                    compress=2)
        
        self.vq = ResidualVectorQuantizer(dimension=512, 
                                        n_q=1, bins=1024, 
                                        kmeans_iters=800,
                                        decay=0.99, 
                                        kmeans_init=True)

        self.decoder = VocosBackbone(input_channels=512, 
                                    dim=768, 
                                    intermediate_dim=2304, 
                                    num_layers=12)
        
        self.head = ISTFTHead(dim=768, 
                            n_fft=1280, 
                            hop_length=320)

    def forward(self, x, bandwidth=6.6):
        z = self.encoder(x)

        vq_result = self.vq(z, frame_rate=75, bandwidth=bandwidth)

        decoded = self.decoder(vq_result.quantized)
        audio = self.head(decoded)
        if audio.dim() == 2:    
            audio = audio.unsqueeze(1)
        return {
            "audio": audio,
            "commit_loss": vq_result.loss,
            "codes": vq_result.codes if hasattr(vq_result, "codes") else None
        }
    
if __name__ == "__main__":
    # Test the model with a random input
    model = HOA_WavTokenizer()
    x = torch.randn(1, 1, 24000)  # Batch size 1, 1 channel, 24000 samples (1.5 xssecond at 16kHz)
    output = model(x, bandwidth=6.6)
    print(output.shape)  # Expected output shape: (1, 24000)