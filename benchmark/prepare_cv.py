"""Materialize machine-local paths from the portable release split manifests."""
import argparse
import csv
import json
import shutil
from collections import Counter,defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();source=a.dataset.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    names=(source/'classes.txt').read_text().splitlines()
    dataset=out/'dataset'
    if not dataset.exists():dataset.symlink_to(source,target_is_directory=True)
    elif dataset.resolve()!=source:raise RuntimeError('Existing experiment points to another dataset')
    with (Path(__file__).resolve().parent/'manifests/split_manifest.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    for fold in range(5):
        dest=out/'splits'/f'fold_{fold}';dest.mkdir(parents=True,exist_ok=True)
        groups={}
        for role,count in [('train',300),('val',100),('test',100)]:
            subset=[r for r in rows if int(r['fold'])==fold and r['role']==role];assert len(subset)==count
            groups[role]={r['source_template_group'] for r in subset}
            (dest/f'{role}.txt').write_text(''.join(str(dataset/'images'/r['image_name'])+'\n' for r in subset))
        assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['test']&groups['val'])
        (dest/'data.yaml').write_text(f'path: {dataset}\ntrain: {dest}/train.txt\nval: {dest}/val.txt\ntest: {dest}/test.txt\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(names)))
    for name in ['image_manifest.csv','template_groups.csv','split_manifest.csv','split_summary.csv','split_audit.json']:
        shutil.copy2((source/'metadata'/name) if name in ['image_manifest.csv','template_groups.csv'] else (Path(__file__).resolve().parent/'manifests'/name),out/name)
    print(json.dumps({'experiment_root':str(out),'folds':5,'images_per_fold':[300,100,100]},indent=2))

if __name__=='__main__':main()
