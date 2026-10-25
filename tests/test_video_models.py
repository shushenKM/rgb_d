"""Model behavior checks on invented clips only; never read the raw dataset."""
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from vision.backbone import ResNet18Features, temporal_shift
from vision.models import SCALES, build_model, model_info, MultiscaleAttentionBlock


def config(model='tsm', modality='rgb'):
    return dict(model=model, modality=modality, input={'num_frames': 2}, model_parameters={
        'dropout': 0., 'dropout_backbone': 0., 'dropout_transformer': 0., 'dff': 32})


class VideoModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shift_changes_temporal_channels_without_crossing_clip(self):
        x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(6, 4, 1, 1).requires_grad_()
        y = temporal_shift(x, 3, 4).reshape(2, 3, 4)
        original = x.reshape(2, 3, 4)
        self.assertTrue(torch.equal(y[:, :-1, 0], original[:, 1:, 0]))
        self.assertTrue(torch.equal(y[:, 1:, 1], original[:, :-1, 1]))
        self.assertTrue(torch.equal(y[:, :, 2:], original[:, :, 2:]))
        self.assertTrue(torch.equal(y[:, -1, 0], torch.zeros(2)))
        self.assertTrue(torch.equal(y[:, 0, 1], torch.zeros(2)))
        y.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_forward_backward_models_and_modalities(self):
        for kind in ('tsm', 'mvtn'):
            for modality, channels in (('rgb', 3), ('depth', 1)):
                with self.subTest(kind=kind, modality=modality):
                    model = build_model(config(kind, modality), 9)
                    logits = model(torch.randn(2, 1, 2, channels, 32, 32))
                    self.assertEqual(tuple(logits.shape), (2, 9))
                    self.assertTrue(torch.isfinite(logits).all())
                    nn.functional.cross_entropy(logits, torch.tensor([0, 8])).backward()
                    self.assertIsNotNone(model.backbone.conv1.weight.grad)
                    self.assertGreater(float(model.backbone.conv1.weight.grad.abs().sum()), 0)
                    if kind == 'mvtn':
                        self.assertEqual([b.fc_q.out_features for b in model.encoder], list(SCALES))
                        self.assertTrue(all(b.fc_q.weight.grad is not None for b in model.encoder))

    def test_two_views_equal_mean_of_separate_views_for_each_model(self):
        for kind in ('tsm', 'mvtn'):
            with self.subTest(kind=kind):
                model = build_model(config(kind), 9).eval()
                x = torch.randn(2, 2, 2, 3, 32, 32)
                with torch.no_grad():
                    joint = model(x)
                    separate = (model(x[:, :1]) + model(x[:, 1:])) / 2
                torch.testing.assert_close(joint, separate, rtol=1e-5, atol=1e-6)

    def test_attention_matches_upstream_single_residual_order(self):
        block = MultiscaleAttentionBlock(16, 8, 32, 0.).eval()
        x = torch.randn(2, 3, 512)
        q = block.fc_q(x).view(2, 3, 8, 2).permute(0, 2, 1, 3)
        k = block.fc_k(x).view(2, 3, 8, 2).permute(0, 2, 3, 1)
        v = block.fc_v(x).view(2, 3, 8, 2).permute(0, 2, 1, 3)
        z = (torch.softmax(q @ k / 2 ** .5, dim=-1) @ v).permute(0, 2, 1, 3).reshape(2, 3, 16)
        expected = block.layer_norm(x + block.fc(block.fc_o(z)))
        torch.testing.assert_close(block(x), expected)

    def test_local_rgb_checkpoint_adapts_depth_and_rejects_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'backbone.pt'
            state = ResNet18Features().state_dict()
            torch.save(state, path)
            settings = config('tsm', 'depth')
            settings['model_parameters']['pretrained_backbone'] = str(path)
            model = build_model(settings, 9)
            torch.testing.assert_close(model.backbone.conv1.weight, state['conv1.weight'].mean(1, keepdim=True))
            self.assertEqual(model_info(settings)['initialization'], 'local_backbone_checkpoint')
            torch.save({'conv1.weight': state['conv1.weight']}, path)
            with self.assertRaisesRegex(ValueError, 'complete ResNet18'):
                build_model(settings, 9)

    def test_invalid_config_and_input(self):
        for change in ({'model': 'other'}, {'modality': 'flow'}, {'input': {'num_frames': 1}},
                       {'model_parameters': {'n_head': 3}}):
            settings = config()
            settings.update(change)
            with self.assertRaises(ValueError):
                build_model(settings, 9)
        model = build_model(config(), 9)
        for shape in ((1, 2, 3, 32, 32), (0, 1, 2, 3, 32, 32), (1, 1, 3, 3, 32, 32),
                      (1, 1, 2, 1, 32, 32), (1, 1, 2, 3, 16, 16)):
            with self.assertRaises(ValueError):
                model(torch.randn(shape))


if __name__ == '__main__':
    unittest.main()
