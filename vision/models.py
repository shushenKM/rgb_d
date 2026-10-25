"""Build TSM or the supplied Multiscale Video Transformer for RGB/depth clips.

INPUT / OUTPUT
    build_model(config, num_classes) returns a torch module accepting
    [batch, views, frames, channels, height, width] and returning class logits.
    RGB uses three channels; depth uses one. Each view is a complete temporal
    clip processed with shared weights, then its logits are averaged. Temporal
    shifts and attention never mix time steps belonging to different views.

ARCHITECTURES
    TSM: ResNet18 with the upstream blockres temporal shift before every basic
    block's conv1; a dropout/linear classifier and mean temporal consensus.
    MVTN: the supplied ResNet18 frame encoder followed by six attention blocks
    with Q/K/V dimensions 512, 256, 128, 64, 32, 16, sinusoidal positions,
    temporal mean pooling, and a classifier. Its attention->feed-forward->one
    residual LayerNorm ordering is deliberately preserved. This is not a
    replacement with torch.nn.TransformerEncoder. Upstream unused spatial
    embedding and cls_token parameters are omitted because forward ignores them.

PORTABILITY / WEIGHTS
    All tensors follow the input device; no fixed CUDA indices are used.
    Only torch is needed. Initialization is random unless the user supplies a
    local standard ResNet18 backbone checkpoint in model_parameters.
    pretrained_backbone. Relative checkpoint paths resolve from this repository.
    No downloads occur. RGB ImageNet conv1 weights can be averaged for depth,
    as in the supplied MVTN implementation. Other shape/key mismatches fail.
    Upstream model files are never edited or imported through training scripts.
    model_info() records source hashes and configuration for reproducibility.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from vision.backbone import ResNet18Features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALES = (512, 256, 128, 64, 32, 16)
UPSTREAM_SOURCES = {
    'tsm': ['models/tsm/ops/temporal_shift.py', 'models/tsm/ops/models.py',
            'models/mvtn/src_mvtn/models/backbones/resnet.py'],
    'mvtn': ['models/mvtn/src_mvtn/models/temporal.py', 'models/mvtn/src_mvtn/models/attention.py',
             'models/mvtn/src_mvtn/models/backbones/resnet.py'],
}
DEFAULT_PARAMETERS = dict(shift_div=8, dropout=0.5, dropout_backbone=0.1,
                          dropout_transformer=0.5, n_head=8, dff=1024,
                          pretrained_backbone=None)


def _settings(config):
    kind = config.get('model', 'tsm')
    if kind not in ('tsm', 'mvtn'):
        raise ValueError("model must be 'tsm' or 'mvtn'")
    modality = config.get('modality', 'rgb')
    if modality not in ('rgb', 'depth'):
        raise ValueError("modality must be 'rgb' or 'depth'")
    frames = config.get('input', {}).get('num_frames', 8)
    if type(frames) is not int or frames < 2:
        raise ValueError('input.num_frames must be an integer of at least two')
    supplied = config.get('model_parameters') or {}
    unknown = set(supplied).difference(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError('Unknown model_parameters: ' + ', '.join(sorted(unknown)))
    params = dict(DEFAULT_PARAMETERS, **supplied)
    for key in ('dropout', 'dropout_backbone', 'dropout_transformer'):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(key + ' must be finite in [0,1)')
    if type(params['shift_div']) is not int or not 2 <= params['shift_div'] <= 64:
        raise ValueError('shift_div must be an integer between 2 and 64')
    if type(params['n_head']) is not int or params['n_head'] < 1 or any(s % params['n_head'] for s in SCALES):
        raise ValueError('n_head must divide every MVTN scale (512,256,128,64,32,16)')
    if type(params['dff']) is not int or params['dff'] < 1:
        raise ValueError('dff must be a positive integer')
    if params['pretrained_backbone'] is not None and (not isinstance(params['pretrained_backbone'], str) or not params['pretrained_backbone'].strip()):
        raise ValueError('pretrained_backbone must be null or a local checkpoint path')
    return kind, modality, frames, params


def _checkpoint_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _load_backbone(backbone, value):
    if value is None:
        return
    path = _checkpoint_path(value)
    state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict) or not state:
        raise ValueError('Backbone checkpoint must contain a nonempty ResNet18 state dictionary')
    state = {key.removeprefix('module.'): tensor for key, tensor in state.items()}
    state.pop('fc.weight', None)
    state.pop('fc.bias', None)
    if (backbone.conv1.in_channels == 1 and 'conv1.weight' in state
            and isinstance(state['conv1.weight'], torch.Tensor)
            and state['conv1.weight'].ndim == 4 and state['conv1.weight'].shape[1] == 3):
        state['conv1.weight'] = state['conv1.weight'].mean(dim=1, keepdim=True)
    try:
        backbone.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError('Checkpoint is not a compatible complete ResNet18 backbone: ' + str(error)) from error


class MultiscaleAttentionBlock(nn.Module):
    """The exact attention and feed-forward ordering used by supplied MVTN."""

    def __init__(self, scale, heads, dff, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = scale // heads
        self.fc_q = nn.Linear(512, scale)
        self.fc_k = nn.Linear(512, scale)
        self.fc_v = nn.Linear(512, scale)
        self.fc_o = nn.Linear(scale, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(512, dff), nn.ReLU(inplace=True),
                                nn.Dropout(dropout), nn.Linear(dff, 512))
        self.layer_norm = nn.LayerNorm(512)
        for layer in (self.fc_q, self.fc_k, self.fc_v, self.fc_o):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        batch, length, _ = x.shape
        q = self.fc_q(x).reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.fc_k(x).reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.fc_v(x).reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        weights = torch.softmax(torch.matmul(q, k) / math.sqrt(self.head_dim), dim=-1)
        attended = torch.matmul(weights, v).permute(0, 2, 1, 3).contiguous().reshape(batch, length, -1)
        attended = self.dropout(self.fc_o(attended))
        attended = self.dropout(self.fc(attended))
        return self.layer_norm(x + attended)


def sinusoidal_positions(length, width=512):
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(width // 2, dtype=torch.float32).unsqueeze(0)
    phase = positions / (10000 ** (2 * dim / width))
    out = torch.empty(length, width)
    out[:, ::2] = torch.sin(phase)
    out[:, 1::2] = torch.cos(phase)
    return out


class VideoRecognitionModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError('num_classes must be an integer of at least two')
        self.kind, self.modality, self.num_frames, params = _settings(config)
        self.in_channels = 3 if self.modality == 'rgb' else 1
        self.backbone = ResNet18Features(
            self.in_channels, dropout=params['dropout_backbone'] if self.kind == 'mvtn' else 0.,
            num_frames=self.num_frames if self.kind == 'tsm' else None, shift_div=params['shift_div'])
        _load_backbone(self.backbone, params['pretrained_backbone'])
        if self.kind == 'mvtn':
            self.encoder = nn.ModuleList([MultiscaleAttentionBlock(scale, params['n_head'], params['dff'],
                params['dropout_transformer']) for scale in SCALES])
            self.register_buffer('positions', sinusoidal_positions(self.num_frames).unsqueeze(0), persistent=False)
            self.classifier = nn.Linear(512, num_classes)
        else:
            self.dropout = nn.Dropout(params['dropout'])
            self.classifier = nn.Linear(512, num_classes)
            nn.init.normal_(self.classifier.weight, 0., .001)
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        if x.ndim != 6:
            raise ValueError('Video input must have shape [B,V,T,C,H,W]')
        batch, views, frames, channels, height, width = x.shape
        if batch < 1 or views < 1 or frames != self.num_frames or channels != self.in_channels or min(height, width) < 32:
            raise ValueError('Invalid video input: require positive batch/views, configured frames/channels and H,W >= 32')
        if not torch.is_floating_point(x):
            raise ValueError('Video model inputs must be floating point, normalized by the loader')
        # Contiguous B,V,T order makes every view an independent T-frame clip.
        features = self.backbone(x.reshape(batch * views * frames, channels, height, width))
        features = features.reshape(batch * views, frames, 512)
        if self.kind == 'mvtn':
            features = features + self.positions.to(dtype=features.dtype)
            for block in self.encoder:
                features = block(features)
            logits = self.classifier(features.mean(dim=1))
        else:
            logits = self.classifier(self.dropout(features)).mean(dim=1)
        return logits.reshape(batch, views, -1).mean(dim=1)


def build_model(config, num_classes):
    """Construct an independent model with raw logits suitable for cross entropy."""
    return VideoRecognitionModel(config, num_classes)


def model_info(config):
    """Return architecture, initialization and original-source provenance."""
    kind, modality, frames, params = _settings(config)
    sources = {}
    for name in UPSTREAM_SOURCES[kind]:
        path = PROJECT_ROOT / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    weights = params['pretrained_backbone']
    return dict(model=kind, modality=modality, backbone='resnet18', num_frames=frames,
                input_layout='B,V,T,C,H,W', view_fusion='mean_logits_shared_weights',
                model_parameters=params, initialization='local_backbone_checkpoint' if weights else 'random',
                pretrained_backbone_sha256=hashlib.sha256(_checkpoint_path(weights).read_bytes()).hexdigest() if weights else None,
                multiscale_dimensions=list(SCALES) if kind == 'mvtn' else None,
                upstream_source_sha256=sources,
                adaptations=['portable device/dtype placement', 'RGB/depth input channels',
                             'shared per-view model and mean-logit fusion', 'independent local training pipeline',
                             'inactive upstream spatial_embedding/cls_token omitted' if kind == 'mvtn'
                             else 'ResNet18 blockres shifts; all BatchNorm layers trainable'])
