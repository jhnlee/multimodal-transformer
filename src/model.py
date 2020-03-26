import logging
import math
import torch
from torch import nn
from torch.nn import functional as F
from modules import TransformerEncoder, CrossmodalTransformerEncoder
from fairseq.modules import PositionalEmbedding

logger = logging.getLogger(__name__)


class MULTModel(nn.Module):
    def __init__(
        self,
        do_vision,
        do_audio,
        do_text,
        orig_d_v,
        orig_d_a,
        orig_d_t,
        n_head,
        n_cmlayer,
        n_salayer,
        p_dropout,
        d_model,
        d_out,
        max_position=128,
        attn_mask=None,
        scale_embedding=True,
    ):
        super(MULTModel, self).__init__()

        # Wheter to use whole mode or not
        do = do_vision + do_audio + do_text
        assert do == 3 or do == 1
        self.partial_mode = True if do == 1 else False

        self.do_vision, self.do_audio, self.do_text = do_vision, do_audio, do_text
        self.d_model = d_model
        self.p_dropout = p_dropout
        self.emb_scale = math.sqrt(d_model) if scale_embedding else 1.0

        combined_dim = 2 * d_model if self.partial_mode == 1 else 6 * d_model

        # Input Encoder (Temporal convolution layers) -> (B, orig_d, L) => (B, d, L)
        self.vision_encoder = nn.Conv1d(orig_d_v, d_model, kernel_size=1, padding=0, bias=False)
        self.audio_encoder = nn.Conv1d(orig_d_a, d_model, kernel_size=1, padding=0, bias=False)
        self.text_encoder = nn.Conv1d(orig_d_t, d_model, kernel_size=1, padding=0, bias=False)

        # Positional Encoder for Inputs -> (B, L) => (B, L, d)
        self.vision_pos = PositionalEmbedding(max_position, d_model, 0)
        self.audio_pos = PositionalEmbedding(max_position, d_model, 0)
        self.text_pos = PositionalEmbedding(max_position, d_model, 0)

        # Cross-modal Transformer layers -> (L, B, d)
        if self.do_vision:
            self.vision_layers_with_audio = CrossmodalTransformerEncoder(
                d_model, n_head, 4 * d_model, p_dropout, n_cmlayer
            )
            self.vision_layers_with_text = CrossmodalTransformerEncoder(
                d_model, n_head, 4 * d_model, p_dropout, n_cmlayer
            )

        if self.do_audio:
            self.audio_layers_with_vision = CrossmodalTransformerEncoder(
                d_model, n_head, 4 * d_model, p_dropout, n_cmlayer
            )
            self.audio_layers_with_text = CrossmodalTransformerEncoder(
                d_model, n_head, 4 * d_model, p_dropout, n_cmlayer
            )

        if self.do_text:
            self.text_layers_with_vision = CrossmodalTransformerEncoder(
                d_model, n_head, 4 * d_model, p_dropout, n_cmlayer
            )
            self.text_layers_with_audio = CrossmodalTransformerEncoder(
                d_model, n_head, 4 * d_model, p_dropout, n_cmlayer
            )

        # Self-Attention layers -> (L, B, d)
        self.vision_layers = TransformerEncoder(
            2 * d_model, n_head, 4 * d_model, p_dropout, n_salayer
        )
        self.audio_layers = TransformerEncoder(
            2 * d_model, n_head, 4 * d_model, p_dropout, n_salayer
        )
        self.text_layers = TransformerEncoder(
            2 * d_model, n_head, 4 * d_model, p_dropout, n_salayer
        )

        # Projection layers
        self.fc_layer1 = nn.Linear(combined_dim, combined_dim)
        self.fc_layer2 = nn.Linear(combined_dim, combined_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.out_layer = nn.Linear(combined_dim, d_out)

    def forward(self, x_vision, x_audio, x_text):
        """
            Args:
        x_vision, x_audio, x_text : input tensor -> (B, L, d)
        """
        # (B, L, d) => (B, d, L)
        x_vision = x_vision.transpose(1, 2)
        x_audio = x_audio.transpose(1, 2)
        x_text = F.dropout(x_text.transpose(1, 2), self.p_dropout, self.training)

        # (B, d, L) => (B, L, d)
        x_vision = F.dropout(
            self.vision_encoder(x_vision).transpose(1, 2), self.p_dropout, self.training
        )
        x_audio = F.dropout(
            self.audio_encoder(x_audio).transpose(1, 2), self.p_dropout, self.training
        )
        x_text = F.dropout(self.text_encoder(x_text).transpose(1, 2), self.p_dropout, self.training)

        # Add Positional Encoding
        vis_pos = self.vision_pos(x_vision[:, :, 0])
        aud_pos = self.audio_pos(x_audio[:, :, 0])
        tex_pos = self.text_pos(x_text[:, :, 0])

        x_pos_pair = zip([x_vision, x_audio, x_text], [vis_pos, aud_pos, tex_pos])
        # (B, L, d) => (L, B, d)
        x_vision, x_audio, x_text = [
            (self.emb_scale * x + p).transpose(0, 1) for x, p in x_pos_pair
        ]

        # Crossmodal Attention
        x_whole = []
        if self.do_vision:
            x_vision_with_audio = self.vision_layers_with_audio(x_audio, x_vision)
            x_vision_with_text = self.vision_layers_with_text(x_text, x_vision)
            x_vision2 = torch.cat([x_vision_with_audio, x_vision_with_text], dim=2)
            x_whole.append(self.vision_layers(x_vision2)[-1])  # take it from last time step

        if self.do_audio:
            x_audio_with_vision = self.audio_layers_with_vision(x_vision, x_audio)
            x_audio_with_text = self.audio_layers_with_text(x_text, x_audio)
            x_audio2 = torch.cat([x_audio_with_vision, x_audio_with_text], dim=2)
            x_whole.append(self.audio_layers(x_audio2)[-1])

        if self.do_text:
            x_text_with_vision = self.text_layers_with_vision(x_vision, x_text)
            x_text_with_audio = self.text_layers_with_audio(x_audio, x_text)
            x_text2 = torch.cat([x_vision_with_audio, x_vision_with_text], dim=2)
            x_whole.append(self.text_layers(x_text2)[-1])

        x_whole = x_whole[0] if self.partial_mode else torch.cat(x_whole, dim=1)

        x_whole2 = F.relu(self.fc_layer1(x_whole))
        x_whole2 = self.fc_layer2(F.dropout(x_whole, p=self.p_dropout, training=self.training))
        x_whole = x_whole2 + x_whole

        out = self.out_layer(x_whole)
        return out
