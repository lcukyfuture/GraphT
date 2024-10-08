import torch
from torch import nn
from layer import SimpleTransformerEncoderLayer
from timeit import default_timer as timer

class SimpleTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, pe, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, pe=pe, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class GraphTransformer(nn.Module):
    # This is a variant of the GraphTransformer, where the node positional
    # information is injected in the attention score instead of being
    # added to the node features. This is in the spirit of relative
    # pos encoding rather than Vaswani et al.
    def __init__(self, in_size, nb_class, d_model,
                 dim_feedforward=512, dropout=0.1, nb_layers=4,
                 batch_norm=False):
        super(GraphTransformer, self).__init__()

        self.embedding = nn.Linear(in_features=in_size,
                                   out_features=d_model,
                                   bias=False)
        encoder_layer = SimpleTransformerEncoderLayer(
                d_model, dim_feedforward, dropout, batch_norm=batch_norm)
        self.encoder = SimpleTransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalAvg1D()
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, nb_class)
            )
                     
    def forward(self, x, masks, pe):
        # We permute the batch and sequence following pytorch
        # Transformer convention
        # st = timer()
        x = x.permute(1, 0, 2)
        output = self.embedding(x)
        
        output = self.encoder(output, pe, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)
        # we make sure to correctly take the masks into account when pooling
        output = self.pooling(output, masks)

        return self.classifier(output)

class GlobalAvg1D(nn.Module):
    def __init__(self):
        super(GlobalAvg1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask = (~mask).float().unsqueeze(-1)
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1)
