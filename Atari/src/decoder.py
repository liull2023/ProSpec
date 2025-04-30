import torch
import torch.nn as nn
import torch.nn.functional as F

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}

class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers]
        self.fc = nn.Linear(self.feature_dim, num_filters * out_dim * out_dim)

        conv_modules = []
        for i in range(num_layers - 1):
            conv_modules.append(nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1))
        conv_modules.append(nn.ConvTranspose2d(num_filters, obs_shape[0], 3, stride=2, output_padding=1))
        self.convs = nn.ModuleList(conv_modules)

        self.outputs = dict()

    def forward(self, encoded_features):
        h = self.fc(encoded_features)
        h = h.view(h.size(0), -1, *([int((self.obs_shape[-1] - 2) / 2 ** self.num_layers)] * 2))
        self.outputs['fc'] = h

        for i in range(self.num_layers - 1):
            h = F.relu(self.convs[i](h))
            self.outputs['deconv%s' % (i + 1)] = h

        h = self.convs[-1](h)
        self.outputs['deconv%s' % self.num_layers] = h

        reconstructed_obs = torch.sigmoid(h)
        self.outputs['reconstructed_obs'] = reconstructed_obs

        return reconstructed_obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_decoder/deconv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_decoder/fc', self.fc, step)

class IdentityDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args):
        super().__init__()
        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, encoded_features):
        return encoded_features

    def log(self, L, step, log_freq):
        pass

_AVAILABLE_DECODERS = {
    'pixel': PixelDecoder,
    'identity': IdentityDecoder
}

def make_decoder(decoder_type, obs_shape, feature_dim, num_layers, num_filters):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](obs_shape, feature_dim, num_layers, num_filters)
