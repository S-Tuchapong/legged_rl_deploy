#!/usr/bin/env python3
"""Export the pinned full-body student and verify recurrent ONNX parity on CPU."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from training_core import Student, Teacher, corrupt_axial

ROOT = Path(__file__).resolve().parent
INPUTS = ['actor', 'depth', 'hidden_in', 'visual_cache_in', 'hold_in']
OUTPUTS = ['actions', 'hidden_out', 'visual_cache_out', 'hold_out']


class ExportPool(nn.Module):
    """Exact AdaptiveAvgPool2d(3,5) for the fixed 60x106 convolution output."""
    def forward(self, x):
        return torch.cat([
            torch.cat([
                x[:, :, i*20:(i+1)*20, j*106//5:math.ceil((j+1)*106/5)]
                .mean(dim=(2, 3), keepdim=True) for j in range(5)
            ], dim=3) for i in range(3)
        ], dim=2)


class Deployment(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student
        self.student.visual[4] = ExportPool()

    def forward(self, actor, depth, hidden, visual_cache, hold):
        # RealSense supplies axial metres, whereas MuJoCo supplied ray ranges.
        z = torch.where(depth > 0, depth, torch.full_like(depth, 3.0))
        normalized = (z.clamp(0.3, 3.0) - 0.3) / 2.7
        current = self.student.visual(normalized)
        visual = torch.where(hold.reshape(1, 1) < 0.5, current, visual_cache)
        window = torch.cat([
            part.reshape(1, 61, dim) for part, dim in zip(
                self.student.normalizer(actor).split([183,183,183,122,1769,1769,1769], -1),
                (3,3,3,2,29,29,29))
        ], -1)
        prop = self.student.proprio(window)
        features = torch.cat((visual, window[:, -1], (1-hold)[:,None], (hold*0.02)[:,None]), -1)
        hidden = self.student.terrain(features, hidden)
        action = self.student.actor(torch.cat((prop, hidden), -1))
        return action, hidden, visual, 1-hold


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7000)
    checkpoint = torch.load(ROOT/'model_7000.pt', map_location='cpu', weights_only=False)
    source_sha = hashlib.sha256((ROOT/'training_core.py').read_bytes()).hexdigest()
    assert source_sha == checkpoint['config']['source_hashes'][
        'src/tasks/distillation/depth_memory_fullbody_course_ground_v3_2gpu512/core.py']
    reference = Student(Teacher()).eval()
    reference.load_state_dict(checkpoint['model'], strict=True)
    student = Student(Teacher()).eval()
    student.load_state_dict(checkpoint['model'], strict=True)
    model = Deployment(student).eval()
    args = (torch.zeros(1,5978), torch.ones(1,1,240,424), torch.zeros(1,64),
            torch.zeros(1,64), torch.zeros(1))
    output = ROOT/'policy.onnx'
    torch.onnx.export(model, args, output, input_names=INPUTS, output_names=OUTPUTS,
                      opset_version=17, dynamo=False)
    graph = onnx.load(output)
    onnx.helper.set_model_props(graph, {
        'checkpoint_sha256': hashlib.sha256((ROOT/'model_7000.pt').read_bytes()).hexdigest(),
        'training_core_sha256':source_sha,
        'depth_contract':'axial metres, zero -> far, clip [0.3,3], normalize (z-0.3)/2.7',
        'policy_dt':'0.02', 'depth_dt':'0.04', 'history':'61 term-major frames',
    })
    onnx.checker.check_model(graph)
    onnx.save(graph,output)
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    state = [np.zeros((1,64),np.float32), np.zeros((1,64),np.float32), np.zeros(1,np.float32)]
    hidden = torch.zeros(1,64)
    max_error = 0.
    # Cover recurrent rollouts, zeros/no-hit, clipping endpoints and restart.
    with torch.no_grad():
        for tick in range(80):
            reset = tick in (0,39)
            if reset:
                state = [np.zeros_like(x) for x in state]
                hidden.zero_()
                local_tick = 0
            history = torch.randn(1,5978)*0.2
            depth = torch.rand(1,1,240,424)*5
            depth[:,:,:,::7] = 0
            if tick == 5: depth.zero_()
            fresh = local_tick % 2 == 0
            if fresh: cached = corrupt_axial(depth)
            expected, hidden = reference(history,cached,torch.tensor([float(fresh)]),
                torch.tensor([0. if fresh else .02]),hidden,torch.tensor([reset]))
            actual = session.run(OUTPUTS, dict(zip(INPUTS,[history.numpy(),depth.numpy(),*state])))
            for result, wanted in zip(actual[:2], (expected.numpy(),hidden.numpy())):
                max_error = max(max_error,float(np.max(np.abs(result-wanted))))
                np.testing.assert_allclose(result,wanted,rtol=2e-4,atol=2e-5)
            assert actual[3][0] == float(fresh)
            if not fresh: np.testing.assert_array_equal(actual[2],state[1])
            state = actual[1:]
            local_tick += 1
    report = {'checkpoint_updates':checkpoint['completed_updates'], 'training_core_sha256':source_sha,
              'onnx_parity_steps':80,'max_absolute_error':max_error,
              'inputs':{x.name:x.shape for x in session.get_inputs()},
              'outputs':{x.name:x.shape for x in session.get_outputs()}}
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
