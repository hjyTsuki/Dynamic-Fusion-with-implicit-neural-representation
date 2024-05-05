import torch
from torch import nn
from torch.nn import functional as F
import numbers
from einops import rearrange
from mlp import INR


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self,
                 inp_dim,
                 out_dim
                 ):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=inp_dim * 2, oup=inp_dim * 2, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=inp_dim * 2, oup=inp_dim, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=inp_dim, oup=out_dim, expand_ratio=2)

    def forward(self, xr, xt):
        x = torch.cat([xr, xt], dim=1)
        x = self.theta_phi(x)
        x = self.theta_rho(x)
        x = self.theta_eta(x)
        return x


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, BasicConv=BasicConv):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = BasicConv(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, bias=bias,
                                relu=False, groups=hidden_features * 2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, BasicConv=BasicConv):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = BasicConv(dim * 3, dim * 3, kernel_size=3, stride=1, bias=bias, relu=False, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, BasicConv=BasicConv):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, BasicConv=BasicConv)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, BasicConv=BasicConv)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Fusion(nn.Module):
    def __init__(self, in_dim=32):
        super(Fusion, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        x_q = self.query_conv(x)
        y_k = self.key_conv(y)
        energy = x_q * y_k
        attention = self.sig(energy)
        attention_x = x * attention
        attention_y = y * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        y_gamma = self.gamma2(torch.cat((y, attention_y), dim=1))
        y_out = y * y_gamma[:, [0], :, :] + attention_y * y_gamma[:, [1], :, :]

        x_s = x_out + y_out

        return x_s


class MultiscaleNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 3, 3],
                 heads=[1, 2, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(MultiscaleNet, self).__init__()

        self.patch_embed_max = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_max1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_max2 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_max3 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_max = Downsample(dim)
        self.down1_2_max2 = Downsample(dim)
        self.down1_2_max3 = Downsample(dim)
        self.encoder_level2_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_max = Downsample(int(dim * 2 ** 1))
        self.down2_3_max2 = Downsample(int(dim * 2 ** 1))
        self.down2_3_max3 = Downsample(int(dim * 2 ** 1))
        self.latent_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.latent_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.latent_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2_max = Upsample(int(dim * 2 ** 2))
        self.up3_2_max2 = Upsample(int(dim * 2 ** 2))
        self.up3_2_max3 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_max1 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level2_max2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level2_max3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.decoder_level2_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_max = Upsample(int(dim * 2 ** 1))
        self.up2_1_max2 = Upsample(int(dim * 2 ** 1))
        self.up2_1_max3 = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level1_max1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level1_max2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level1_max3 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        self.decoder_level1_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_max2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1_max3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output_max = nn.Conv2d(int(dim * 1 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_max_context1 = nn.Conv2d(int(dim * 1 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_max_context2 = nn.Conv2d(int(dim * 1 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        outputs = list()

        inp_enc_level1_max = self.patch_embed_max(inp_img)
        out_enc_level1_max = self.encoder_level1_max1(inp_enc_level1_max)

        inp_enc_level2_max = self.down1_2_max(out_enc_level1_max)
        out_enc_level2_max = self.encoder_level2_max1(inp_enc_level2_max)

        inp_enc_level4_max = self.down2_3_max(out_enc_level2_max)
        latent_max = self.latent_max1(inp_enc_level4_max)

        inp_dec_level2_max = self.up3_2_max(latent_max)
        inp_dec_level2_max = torch.cat([inp_dec_level2_max, out_enc_level2_max], 1)
        inp_dec_level2_max = self.reduce_chan_level2_max1(inp_dec_level2_max)
        out_dec_level2_max = self.decoder_level2_max1(inp_dec_level2_max)

        inp_dec_level1_max = self.up2_1_max(out_dec_level2_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        inp_dec_level1_max = self.reduce_chan_level1_max1(inp_dec_level1_max)
        out_dec_level1_max = self.decoder_level1_max1(inp_dec_level1_max)

        out_dec_level1_max = self.output_max_context1(out_dec_level1_max)
        out_enc_level1_max = self.encoder_level1_max2(out_dec_level1_max)

        inp_enc_level2_max = self.down1_2_max2(out_enc_level1_max)
        out_enc_level2_max = self.encoder_level2_max2(inp_enc_level2_max)

        inp_enc_level4_max = self.down2_3_max2(out_enc_level2_max)
        latent_max = self.latent_max2(inp_enc_level4_max)

        inp_dec_level2_max = self.up3_2_max2(latent_max)
        inp_dec_level2_max = torch.cat([inp_dec_level2_max, out_enc_level2_max], 1)
        inp_dec_level2_max = self.reduce_chan_level2_max2(inp_dec_level2_max)
        out_dec_level2_max = self.decoder_level2_max2(inp_dec_level2_max)

        inp_dec_level1_max = self.up2_1_max2(out_dec_level2_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        inp_dec_level1_max = self.reduce_chan_level1_max2(inp_dec_level1_max)
        out_dec_level1_max = self.decoder_level1_max2(inp_dec_level1_max)

        out_dec_level1_max = self.output_max_context2(out_dec_level1_max)
        out_enc_level1_max = self.encoder_level1_max3(out_dec_level1_max)

        inp_enc_level2_max = self.down1_2_max3(out_enc_level1_max)
        out_enc_level2_max = self.encoder_level2_max3(inp_enc_level2_max)

        inp_enc_level4_max = self.down2_3_max3(out_enc_level2_max)
        latent_max = self.latent_max3(inp_enc_level4_max)

        inp_dec_level2_max = self.up3_2_max3(latent_max)

        inp_dec_level2_max = torch.cat([inp_dec_level2_max, out_enc_level2_max], 1)
        inp_dec_level2_max = self.reduce_chan_level2_max3(inp_dec_level2_max)
        out_dec_level2_max = self.decoder_level2_max3(inp_dec_level2_max)

        inp_dec_level1_max = self.up2_1_max3(out_dec_level2_max)
        inp_dec_level1_max = torch.cat([inp_dec_level1_max, out_enc_level1_max], 1)
        inp_dec_level1_max = self.reduce_chan_level1_max2(inp_dec_level1_max)
        out_dec_level1_max = self.decoder_level1_max3(inp_dec_level1_max)

        out_dec_level1_max = self.output_max(out_dec_level1_max)

        outputs.append(out_dec_level1_max)

        return outputs[::-1]


class EncoderTransformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 dim=48,
                 num_blocks=[2, 3, 3],
                 heads=[1, 2, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(EncoderTransformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

    def forward(self, inp_img):
        outputs = list()

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        outputs.append(out_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        outputs.append(out_enc_level2)

        inp_enc_level4 = self.down2_3(out_enc_level2)
        latent = self.latent(inp_enc_level4)
        outputs.append(latent)

        return outputs


class DecoderTransformer(nn.Module):
    def __init__(self,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 3, 3],
                 heads=[1, 2, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(DecoderTransformer, self).__init__()
        self.up3_2_max = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_max1 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_max = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level1_max1 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)
        self.decoder_level1_max1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 1 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.output_max = nn.Conv2d(int(dim * 1 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_max_context1 = nn.Conv2d(int(dim * 1 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_max_context2 = nn.Conv2d(int(dim * 1 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_features):
        inp_dec_level2 = self.up3_2_max(inp_features[-1])
        inp_dec_level2 = torch.cat([inp_dec_level2, inp_features[-2]], 1)
        inp_dec_level2 = self.reduce_chan_level2_max1(inp_dec_level2)
        out_dec_level2 = self.decoder_level2_max1(inp_dec_level2)

        inp_dec_level1 = self.up2_1_max(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, inp_features[-2]], 1)
        inp_dec_level1 = self.reduce_chan_level1_max1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1_max1(inp_dec_level1)

        out_dec_level1 = self.output_max_context1(out_dec_level1)
        return out_dec_level1


class DynamicFusion(nn.Module):
    def __init__(self,
                 inp_chanel,
                 feature_levels=3,
                 ):
        super(DynamicFusion, self).__init__()
        self.feature_levels = feature_levels
        for i in range(feature_levels):
            setattr(self, f'fusion_stage{i}', DetailNode(inp_chanel * (2 ** i), 1))
            setattr(self, f'INR{i}', INR(1).cuda())

    def forward(self, rgb_feature_list, thermal_feature_list):
        fused_list = []
        for i in range(self.feature_levels):
            xr = rgb_feature_list[i]
            xt = thermal_feature_list[i]
            gate = getattr(self, f'fusion_stage{i}')
            dynamic = gate(xr, xt)
            INR_Trans = getattr(self, f'INR{i}')
            dynamic = INR_Trans(dynamic).sigmoid()
            fused = xr * dynamic + xt * (1 - dynamic)
            fused_list.append(fused)
        return fused_list

class Net(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=1,
                 dim=48,
                 num_blocks=[2, 3, 3],
                 heads=[1, 2, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Net, self).__init__()
        self.encoder = EncoderTransformer(
            inp_channels=inp_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.fusion_layers = DynamicFusion(
            inp_chanel=dim,
            feature_levels=len(num_blocks)
        )
        self.decoder = DecoderTransformer(
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

    def forward(self, xr, xt):
        rgb_feature_list = self.encoder(xr)
        thermal_feature_list = self.encoder(xt)

        fusion_feature_list = self.fusion_layers(rgb_feature_list, thermal_feature_list)

        output = self.decoder(fusion_feature_list)
        return output.sigmoid()

if __name__ == '__main__':
    img_rgb, img_thermal = torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224)
    img_rgb = img_rgb.to(torch.device('cuda:0'))
    img_thermal = img_thermal.to(torch.device('cuda:0'))
    model = Net().cuda()
    model(img_rgb, img_thermal)
    print('a')
