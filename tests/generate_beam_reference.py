"""Generate raw ONNX reference outputs for the C++ runtime contract test."""
from pathlib import Path
import hashlib
import numpy as np
import onnxruntime as ort
import yaml

root = Path(__file__).resolve().parents[1]
policy = root / 'policies/g1/beam/g1_beam_depth_distillation'
reference = {}
for variant in ('V1', 'V2', 'V3', 'V4'):
    path = next((policy / variant).glob('*.onnx'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    hidden_shape = next(i.shape for i in session.get_inputs() if i.name == 'hidden')
    hidden = np.zeros(hidden_shape, np.float32)
    steps = []
    for step in range(4):
        if step == 3:
            hidden.fill(0)
        action, hidden = session.run(None, dict(
            history=np.full((1, 5978), .1, np.float32),
            depth=np.full((1, 1, 120, 212), .5, np.float32),
            fresh=np.array([step != 1], np.float32),
            age=np.array([.02 if step == 1 else 0], np.float32),
            hidden=hidden, reset=np.array([step in (0, 3)], bool)))
        steps.append(dict(action=action.ravel().tolist(), hidden=hidden.ravel().tolist()))
    reference[variant] = steps
    reference[variant + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
output = root / 'tests/data/beam_variant_reference.yaml'
output.parent.mkdir(exist_ok=True)
output.write_text(yaml.safe_dump(reference, sort_keys=False))
print(output)
