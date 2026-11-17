#!/usr/bin/env python3
"""Deterministic standalone MuJoCo evaluation of the V1--V4 raw ONNX cores.

No ROS/hardware connections. See output protocol.json for exact scoring/setup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort
import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'policies/g1/beam/g1_beam_depth_distillation'
SCENES = ROOT.parent / 'unitree_mujoco/unitree_robots/g1'
DT = .02
SEED_X_RANGE = (-.2899984035853071, .264001596414693)
PROTOCOL = dict(
    engine='standalone MuJoCo; workspace G1 robot and level XMLs; not ROS deployment',
    command=[.6, 0., 0.], policy_hz=50, physics_hz=1000, pd_hz=200,
    depth='212x120 axial metres / 6, clipped [0,1], invalid=1, 50 Hz, fresh=1 age=0; clean all variants',
    camera=dict(parent='torso_link', pos=[.12,0,.42], quat=[.60483387,.36630041,-.36630041,-.60483387], fovy=58.310943190845826),
    history='61 frames per term, oldest first, repeat first on reset; no observation noise/delay',
    reset=dict(seeds=list(range(10)), generator='numpy.default_rng(seed)', x_uniform=list(SEED_X_RANGE), y=0, yaw=0, z=.793,
               joints='config nominal offsets', velocities=0, settling='none', note='Training platform-only x range translated +0.4 m; common seeds across all models/levels'),
    near='pelvis x >= 4.10 m (0.30 m before far platform), abs(y)<=0.40 m, pelvis z > 0.45 m, upright cos > 0.5, course foot contact within last 0.12 s, no prior ground contact/fall',
    onto='near achieved; pelvis x in [4.40,5.20], abs(y)<=0.40, same upright/support conditions, both feet have contacted platform_b',
    termination='onto, ground contact, pelvis z < 0.40 m, tilt > 60 degrees, abs(y)>1 m, or 15 s',
    scoring='criteria latched independently; near remains success if robot falls before onto',
    video_selection='narrowest selected terrain with 10/10 near; narrowest selected terrain with >0/10 near; lowest successful seed per case; absent categories explicitly reported',
    elastic_support=False, domain_randomization=False,
)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def level_value(value):
    if value == '20cm':
        return value
    if value.isdigit() and 0 <= int(value) <= 9:
        return int(value)
    if value in [f'{i}.5' for i in range(9)]:
        return float(value)
    raise argparse.ArgumentTypeError('terrain must be 0--9, a midpoint 0.5--8.5, or 20cm')


def scene_path(level):
    level_name = str(level).replace('.', 'p')
    name = ('scene_depth_training_width_20cm.xml' if level == '20cm'
            else f'scene_depth_training_level_{level_name}.xml')
    return SCENES / name


def load_model(level):
    spec = mujoco.MjSpec.from_file(str(scene_path(level)))
    spec.body('torso_link').add_camera(name='eval_depth', pos=PROTOCOL['camera']['pos'],
        quat=PROTOCOL['camera']['quat'], fovy=PROTOCOL['camera']['fovy'])
    # 0.001 s divides both the deployed 0.005 s PD and 0.020 s policy periods.
    spec.option.timestep = .001
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    return spec.compile()


class Runner:
    def __init__(self, variant, level):
        self.variant, self.level = variant, level
        self.model = m = load_model(level)
        self.data = mujoco.MjData(m)
        cfg = yaml.safe_load((POLICY/'config.yaml').read_text())['policy']
        self.kp, self.kd = np.array(cfg['stiffness']), np.array(cfg['damping'])
        action = cfg['actions']['JointPositionAction']
        self.scale, self.offset = np.array(action['scale']), np.array(action['offset'])
        clip = cfg['actions']['AppliedActionClip']
        self.low, self.high = np.array(clip['min']), np.array(clip['max'])
        self.path = next((POLICY/variant).glob('*.onnx'))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), opts, providers=['CPUExecutionProvider'])
        inputs = {i.name:i.shape for i in self.session.get_inputs()}
        assert inputs['history'] == [1,5978] and inputs['depth'] == [1,1,120,212], inputs
        self.hidden_size = inputs['hidden'][1]
        # Explicit actuator ordering agrees with config and the training checkpoint.
        self.joints = m.actuator_trnid[:,0]
        self.qadr = m.jnt_qposadr[self.joints]
        self.vadr = m.jnt_dofadr[self.joints]
        assert m.nu == 29 and np.array_equal(self.qadr, np.arange(7,36))
        self.feet = {m.body(f'{s}_ankle_roll_link').id:s for s in ('left','right')}
        self.floor = m.geom('floor').id
        self.platform = m.geom('platform_b').id
        self.course = {m.geom(n).id for n in ('platform_a','straight_bridge','platform_b')}
        self.renderer = mujoco.Renderer(m, height=120, width=212)
        self.renderer.enable_depth_rendering()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1,1,1,0,1,0]

    def run(self, seed):
        m,d = self.model,self.data
        mujoco.mj_resetData(m,d)
        d.qpos[0] = np.random.default_rng(seed).uniform(*SEED_X_RANGE)
        d.qpos[2] = .793
        d.qpos[self.qadr] = self.offset
        mujoco.mj_forward(m,d)
        hidden = np.zeros((1,self.hidden_size), np.float32)
        prev = np.zeros(29, np.float32)
        history = None
        trajectory, times = [d.qpos.copy()], [0.]
        near, onto, touched = False,False,set()
        near_time = onto_time = None
        last_support = -1.
        max_x = float(d.qpos[0])
        reason = 'timeout'
        start = time.monotonic()
        for tick in range(750):
            rot = d.xmat[m.body('pelvis').id].reshape(3,3)
            phase = 2*np.pi*(tick*DT/.9 % 1)
            obs = [np.clip(d.sensor('imu_gyro').data,-10,10)*.25,
                   rot.T @ np.array([0.,0.,-1.]), np.array([.6,0.,0.]),
                   np.array([np.sin(phase),np.cos(phase)]),
                   np.clip(d.qpos[self.qadr]-self.offset,-2,2),
                   np.clip(d.qvel[self.vadr],-30,30)*.05, prev]
            if history is None:
                history = [np.repeat(x[None],61,axis=0).astype(np.float32) for x in obs]
            else:
                for h,x in zip(history,obs):
                    h[:-1] = h[1:]
                    h[-1] = x
            self.renderer.update_scene(d,camera='eval_depth',scene_option=self.opt)
            depth = self.renderer.render()
            depth = np.where(np.isfinite(depth)&(depth>0),depth,6.)
            depth = np.clip(depth/6.,0,1).astype(np.float32)[None,None]
            action,hidden = self.session.run(['action_raw','hidden_next'],dict(
                history=np.concatenate([h.ravel() for h in history])[None], depth=depth,
                fresh=np.ones(1,np.float32), age=np.zeros(1,np.float32), hidden=hidden,
                reset=np.array([tick==0],dtype=bool)))
            assert np.isfinite(action).all() and np.isfinite(hidden).all()
            prev = np.clip(action[0],self.low,self.high).astype(np.float32)
            target = self.offset + self.scale*prev
            failed = False
            for sub in range(20):
                if sub%5 == 0:
                    d.ctrl[:] = self.kp*(target-d.qpos[self.qadr])-self.kd*d.qvel[self.vadr]
                mujoco.mj_step(m,d)
                for contact in d.contact:
                    a,b = int(contact.geom1),int(contact.geom2)
                    if contact.dist > .001:
                        continue
                    if self.floor in (a,b):
                        reason,failed = 'ground_contact',True
                    for terrain,robot in ((a,b),(b,a)):
                        side = self.feet.get(int(m.geom_bodyid[robot]))
                        if terrain in self.course and side:
                            last_support = d.time
                        if terrain == self.platform and side:
                            touched.add(side)
                x,y,z = d.qpos[:3]
                upright = d.xmat[m.body('pelvis').id].reshape(3,3)[2,2] > .5
                if z < .4 or not upright or abs(y)>1:
                    reason,failed = 'fall',True
                if failed:
                    break
                supported = d.time-last_support <= .12
                if x>=4.1 and z>.45 and upright and supported and abs(y)<=.4:
                    if not near: near_time = float(d.time)
                    near=True
                if near and 4.4<=x<=5.2 and abs(y)<=.4 and z>.45 and upright and supported and len(touched)==2:
                    onto,onto_time,reason = True,float(d.time),'onto'
                    break
            max_x = max(max_x,float(d.qpos[0]))
            trajectory.append(d.qpos.copy())
            times.append(float(d.time))
            if failed or onto: break
        result = dict(variant=self.variant, level=self.level, width_m=float(m.geom_size[m.geom("straight_bridge").id, 1]*2),
            seed=seed, near=near, onto=onto, near_time=near_time, onto_time=onto_time,
            reason=reason, duration_s=float(d.time), max_x=max_x, final_xyz=d.qpos[:3].tolist(),
            start_x=float(trajectory[0][0]), wall_s=time.monotonic()-start)
        return result,np.asarray(trajectory),np.asarray(times)

    def close(self):
        self.renderer.close()


def summarize(out, videos, levels=range(10)):
    rows = [json.loads(p.read_text()) for p in sorted((out/'trials').glob('*.json'))]
    summary,selected = [],[]
    for v in ('V1','V2','V3','V4'):
        for level in levels:
            trials = [r for r in rows if r['variant']==v and r['level']==level]
            if len(trials)!=10 or {r['seed'] for r in trials} != set(range(10)):
                raise RuntimeError(f'{v} L{level}: expected exactly seeds 0--9, got {[r["seed"] for r in trials]}')
            summary.append(dict(variant=v,level=level,width_m=trials[0]['width_m'],n=10,
                                near=sum(r['near'] for r in trials),onto=sum(r['onto'] for r in trials)))
        for case,predicate in [('hardest_100pct',lambda n:n==10),('hardest_any_success',lambda n:n>0)]:
            eligible=[r for r in summary if r['variant']==v and predicate(r['near'])]
            if not eligible:
                selected.append(dict(variant=v,case=case,level=None,reason='No qualifying level'))
                continue
            level=min(eligible, key=lambda r:r['width_m'])['level']
            chosen=min((r for r in rows if r['variant']==v and r['level']==level and r['near']),key=lambda r:r['seed'])
            item=dict(variant=v,case=case,level=level,seed=chosen['seed'],near=chosen['near'],onto=chosen['onto'])
            item['video']=f'videos/{v}_{case}_L{level}_seed{chosen["seed"]}.mp4'
            selected.append(item)
            if videos: render_video(out,chosen,out/item['video'],case)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'video_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    import csv
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    print(json.dumps(selected,indent=2),flush=True)


def render_video(out,row,path,case):
    from PIL import Image,ImageDraw
    path.parent.mkdir(exist_ok=True)
    m=load_model(row['level']);d=mujoco.MjData(m)
    traj=np.load(out/'trials'/f'{row["variant"]}_L{row["level"]}_seed{row["seed"]}.npz')
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=[2.4,0,.35];camera.distance=7.3;camera.azimuth=115;camera.elevation=-22
    opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,1,0]
    with mujoco.Renderer(m,height=720,width=1280) as renderer, imageio.get_writer(str(path),fps=50,codec='libx264',quality=8,macro_block_size=1) as writer:
        for q,t in zip(traj['qpos'],traj['time']):
            d.qpos[:]=q;mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=camera,scene_option=opt)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,1280,86),fill=(18,23,30))
            near=row['near_time'] is not None and t>=row['near_time']
            onto=row['onto_time'] is not None and t>=row['onto_time']
            draw.text((20,10),f'{row["variant"]} | {case} | level {row["level"]} | width {row["width_m"]:.3f} m | seed {row["seed"]}',fill='white',font_size=22)
            draw.text((20,45),f'Command (0.60, 0, 0) | t={t:.2f}s | Near: {"YES" if near else "no"} | Onto: {"YES" if onto else "no"}',fill='white',font_size=22)
            writer.append_data(np.asarray(im))
    print(f'VIDEO {path}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--variant',choices=['V1','V2','V3','V4'])
    ap.add_argument('--levels',type=level_value,nargs='+',default=list(range(10)), help='0--9, midpoints 0.5--8.5, or custom 20cm')
    ap.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    ap.add_argument('--summarize',action='store_true')
    ap.add_argument('--videos',action='store_true')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.summarize: summarize(out,args.videos,args.levels);return
    if not args.variant: ap.error('--variant required for trials')
    (out/'trials').mkdir(exist_ok=True)
    protocol={**PROTOCOL,'selected_terrains':args.levels,'selected_seeds':args.seeds,'physics_hz':1000,'mujoco_version':mujoco.__version__,'onnxruntime_version':ort.__version__}
    (out/f'protocol_{args.variant}.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for level in args.levels:
        runner=Runner(args.variant,level)
        for seed in args.seeds:
            stem=out/'trials'/f'{args.variant}_L{level}_seed{seed}'
            result_path = Path(str(stem) + '.json')
            trajectory_path = Path(str(stem) + '.npz')
            if result_path.exists(): continue
            result,qpos,times=runner.run(seed)
            result.update(model_sha256=sha(runner.path),scene_sha256=sha(scene_path(level)),robot_sha256=sha(SCENES/'g1_29dof.xml'))
            np.savez_compressed(trajectory_path,qpos=qpos,time=times)
            result_path.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
        runner.close()

if __name__=='__main__':main()
