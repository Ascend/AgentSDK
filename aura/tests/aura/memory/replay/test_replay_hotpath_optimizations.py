"""Canonical persistence and bit-exact sampler compatibility for P5 optimizations."""
from dataclasses import replace
import importlib.util
import json,random
from pathlib import Path
from types import SimpleNamespace
import pytest
from aura.memory.replay.codec import ReplayCodec,canonical,digest,strict_json
from aura.memory.replay.json_store import JsonReplayStore
from aura.memory.replay.schema import ReplayTensor
from aura.memory.replay.sampling import draw_groups

REFERENCE=json.loads((Path(__file__).parent/'fixtures/priority_hotpath_v1.json').read_text())
_spec=importlib.util.spec_from_file_location('frozen_sampling_reference_v1',Path(__file__).parent/'fixtures/sampling_reference_v1.py')
_legacy=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
@pytest.mark.parametrize('case',REFERENCE['cases'],ids=lambda c:c['name'])
def test_sampling_preserves_indices_exact_probabilities_and_rng_state(case):
 groups=[SimpleNamespace(samples=(SimpleNamespace(source_store_id='store',group_id='g'+str(i),reward=(i%5)*.2),SimpleNamespace(source_store_id='store',group_id='g'+str(i),reward=(i%3)*.3))) for i in range(len(case['priorities']))]
 cfg=SimpleNamespace(replay_strategy=case['strategy'],priority_alpha=case['alpha'],priority_epsilon=case['epsilon'],reward_temperature=case['temperature'])
 priorities={('store','g'+str(i)):v for i,v in enumerate(case['priorities'])}
 rng=random.Random(42)
 legacy_rng=random.Random(42)
 for expected in case['expected']:
  chosen,probs=draw_groups(groups,case['count'],rng,cfg,priorities)
  old_chosen,old_probs=_legacy.draw_groups(groups,case['count'],legacy_rng,cfg,priorities)
  assert [int(g.samples[0].group_id[1:]) for g in chosen]==expected['indices']
  assert chosen==old_chosen and probs==old_probs
  # Cross-platform libm may round a last bit differently than the ARM fixture;
  # exact same-platform legacy equality above remains mandatory.
  assert list(probs)==pytest.approx(expected['probabilities'],rel=1e-14,abs=1e-15)
  assert rng.getstate()==legacy_rng.getstate()
 assert json.loads(json.dumps(rng.getstate()))==case['final_rng_state']

@pytest.mark.parametrize('dtype',['bool','uint8','int32','int64','float16','float32','float64'])
def test_single_body_encoding_matches_existing_codec_bytes(make_group,dtype):
 extra=ReplayTensor.from_values([0,1],dtype)
 original=replace(make_group()[0],reward=-0.,prompt_id='引用"\\\n汉字',payload={**make_group()[0].payload,'extra':extra},created_seq=19,group_complete=True)
 snapshot,encoded,checksum=ReplayCodec.prepare(original)
 assert encoded==ReplayCodec.encode(original)
 assert snapshot==ReplayCodec.decode(encoded)
 assert checksum==strict_json(encoded)['checksum']
 assert original.created_seq==19 and snapshot.created_seq is None
 assert snapshot.payload['responses'] is not original.payload['responses']

def test_json_line_exact_canonical_envelope_and_reopen(tmp_path,make_group):
 p=tmp_path/'journal';store=JsonReplayStore(p,store_id='store')
 samples=[replace(s,prompt_id='引用"\\\n汉字') for s in make_group()]
 receipt=store.append_trajectories(samples,'request')
 store.close()
 for line in p.read_bytes().splitlines(keepends=True):
  value=strict_json(line)
  assert line==canonical({'record':value['record'],'checksum':digest(value['record'])})+b'\n'
 reader=JsonReplayStore(p,read_only=True)
 try:
  assert reader.get_trajectory(samples[0].trajectory_id).old_log_probs.data==samples[0].old_log_probs.data
 finally:reader.close()

def test_priority_logit_memo_is_per_call_and_validates_each_value():
 groups=[SimpleNamespace(samples=(SimpleNamespace(source_store_id='s',group_id=str(i),reward=0.),)) for i in range(2)]
 config=SimpleNamespace(replay_strategy='priority',priority_epsilon=1e-6,priority_alpha=1.,reward_temperature=1.)
 rng=SimpleNamespace(random=lambda:0.)
 priorities={('s','0'):1.,('s','1'):2.}
 _,a=draw_groups(groups,1,rng,config,priorities)
 priorities[('s','1')]=100.
 _,b=draw_groups(groups,1,rng,config,priorities)
 assert a!=b and b[0]<a[0]
 config.priority_epsilon=10.
 _,c=draw_groups(groups,1,rng,config,priorities)
 assert c!=b
 priorities[('s','1')]=float('nan')
 with pytest.raises(ValueError):draw_groups(groups,1,rng,config,priorities)


@pytest.mark.parametrize('size,count', [(7,7),(1024,2),(1024,32),(1024,1024)])
@pytest.mark.parametrize('strategy', ['priority','reward_weighted'])
def test_parallel_sampler_lists_preserve_large_window_and_exhaustion(size,count,strategy):
 groups=[SimpleNamespace(samples=(SimpleNamespace(source_store_id='s',group_id=str(i),reward=(i%7)*.3),))
         for i in range(size)]
 cfg=SimpleNamespace(replay_strategy=strategy,priority_alpha=4.,priority_epsilon=1e-300,reward_temperature=.1)
 values=(0.,1e-300,.1,1.,10.,1e100)
 priorities={('s',str(i)):values[i%len(values)] for i in range(size)}
 rng=random.Random(42);old_rng=random.Random(42)
 original=tuple(groups)
 for _ in range(3):
  actual=draw_groups(groups,count,rng,cfg,priorities)
  expected=_legacy.draw_groups(groups,count,old_rng,cfg,priorities)
  assert actual==expected and rng.getstate()==old_rng.getstate()
  assert tuple(groups)==original


@pytest.mark.parametrize('mask_dtype',['bool','uint8','int32','int64'])
@pytest.mark.parametrize('reward_dtype',['float16','float32','float64'])
@pytest.mark.parametrize('pad',[-7,0])
def test_numpy_repadding_matches_frozen_torch_bytes_and_owns_output(make_group,mask_dtype,reward_dtype,pad):
 import torch
 from aura.memory.replay.conversion import training_batch
 from aura.memory.replay.schema import ReplayGroup
 spec=importlib.util.spec_from_file_location('conversion_reference_v1',Path(__file__).parent/'fixtures/conversion_reference_v1.py')
 reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
 groups=[]
 for g in range(2):
  members=[]
  for i,sample in enumerate(make_group(group_id='g'+str(g))):
   n=(1,3,7,2)[g*2+i]
   rewards=[-0.]*(n-1)+[.12345]
   members.append(replace(sample,payload={
    'prompts':ReplayTensor.from_values([0,5,0,6][:1+g*3],'int32' if g else 'int64'),
    'responses':ReplayTensor.from_values(range(n),'int32' if i else 'int64'),
    'response_mask':ReplayTensor.from_values([1]+[j%2 for j in range(n-1)],mask_dtype),
    'rm_scores':ReplayTensor.from_values(rewards,reward_dtype),
    'token_level_rewards':ReplayTensor.from_values(rewards,reward_dtype)},
    old_log_probs=ReplayTensor.from_values([-.125]*n,'float32')))
  groups.append(ReplayGroup(tuple(members)))
 before=tuple(ReplayCodec.encode(s) for g in groups for s in g.samples)
 actual=training_batch(groups,pad);expected=reference.training_batch(groups,pad)
 assert actual.keys()==expected.keys()
 for key,value in expected.items():
  if isinstance(value,torch.Tensor):
   assert actual[key].dtype==value.dtype and actual[key].shape==value.shape
   assert torch.equal(actual[key].view(torch.uint8),value.view(torch.uint8)),key
  else:assert actual[key]==value
 actual['prompts'].fill_(99);actual['rollout_log_probs'].fill_(99)
 assert tuple(ReplayCodec.encode(s) for g in groups for s in g.samples)==before
 again=training_batch(groups,pad)
 assert torch.equal(again['prompts'],expected['prompts'])
 assert torch.equal(again['rollout_log_probs'],expected['rollout_log_probs'])
