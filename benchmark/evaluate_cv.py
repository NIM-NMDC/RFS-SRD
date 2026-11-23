"""Evaluate validation-selected operating points on held-out test folds."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import json
import math
import numpy as np
import metrics as m

SEEDS=[20261112,20261113,20261114]
MODELS={
    'fasterrcnn_r50_fpn_v2':'Faster R-CNN',
    'retinanet_r50_fpn_v2':'RetinaNet',
    'fcos_r50_fpn':'FCOS',
    'rtdetr_l':'RT-DETR-L',
    'yolo11m':'YOLO11m',
    'yolov10m':'YOLOv10m',
}
FIELDS=['precision','recall','f1','ap50','ap50_95','miss_rate','false_discovery_rate','fppi']

def csv_write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_list(path):return [Path(x).name for x in path.read_text().splitlines() if x.strip()]

def canonical_predictions(predictions, width=2048, height=2048):
    """Apply identical image bounds and candidate limits to all frameworks."""
    result={}
    for name,records in predictions.items():
        candidates=[]
        for record in records:
            c=int(record['class_id']);score=float(record['score'])
            assert 0<=c<6 and math.isfinite(score) and 0<=score<=1
            box=np.asarray(record['xyxy'],dtype=float)
            assert box.shape==(4,) and np.isfinite(box).all()
            if score<0.001:continue
            box=np.clip(box,[0,0,0,0],[width,height,width,height])
            if box[2]<=box[0] or box[3]<=box[1]:continue
            candidates.append({'class_id':c,'score':score,'xyxy':box.tolist()})
        result[name]=sorted(candidates,key=lambda r:r['score'],reverse=True)[:100]
    return result

def select_threshold(gt,pred):
    stats=[m.image_match_stats(gt,pred,c,0.5) for c in range(6)]
    packed=[]
    for s in stats:
        scores=np.concatenate([r['scores'] for r in s.values()])
        tp=np.concatenate([r['tp'] for r in s.values()])
        packed.append((scores,tp,sum(r['n_gt'] for r in s.values())))
    best=-1;chosen=0.5
    for t in np.arange(1,100)/100:
        fs=[]
        for scores,tp,n in packed:
            selected=scores>=t;ntp=int(tp[selected].sum());nselected=int(selected.sum())
            fs.append(2*ntp/(nselected+n) if nselected+n else 0)
        value=float(np.mean(fs))
        if value>best+1e-12 or (abs(value-best)<=1e-12 and t>chosen):best=value;chosen=float(t)
    return chosen,best

def evaluate_job(task):
    rootstr,model,seed,fold=task;root=Path(rootstr)
    run=root/'runs'/model/f'fold_{fold}_seed_{seed}'
    assert (run/'completed.json').is_file(),str(run)
    allgt=m.load_ground_truth(root/'dataset/images',root/'dataset/labels')
    vp,_=m.load_payload(run/'predictions_val.json');tp,_=m.load_payload(run/'predictions_test.json')
    vp=canonical_predictions(vp);tp=canonical_predictions(tp)
    expected_val=set(load_list(root/'splits'/f'fold_{fold}'/'val.txt'))
    expected_test=set(load_list(root/'splits'/f'fold_{fold}'/'test.txt'))
    assert set(vp)==expected_val and set(tp)==expected_test and not(expected_val&expected_test)
    assert len(expected_val)==len(expected_test)==100
    vg={k:allgt[k] for k in sorted(expected_val)};tg={k:allgt[k] for k in sorted(expected_test)}
    threshold,val_f1=select_threshold(vg,vp)
    rows=[]
    for c,name in enumerate(m.CLASS_NAMES):
        stats={iou:m.image_match_stats(tg,tp,c,iou) for iou in m.IOU_THRESHOLDS}
        at50=m.aggregate(stats[0.5],list(tg),threshold)
        aps=[m.aggregate(stats[iou],list(tg),None)['ap'] for iou in m.IOU_THRESHOLDS]
        row={'model':model,'model_label':MODELS[model],'seed':seed,'fold':fold,'class_id':c,'class':name,
             'threshold':threshold,'n_images':100,'n_gt':at50['n_gt'],'tp':at50['tp'],'fp':at50['fp'],'fn':at50['fn'],
             'precision':at50['precision'],'recall':at50['recall'],'f1':at50['f1'],
             'ap50':aps[0],'ap50_95':float(np.mean(aps)),
             'miss_rate':at50['miss_rate'],'false_discovery_rate':at50['false_discovery_rate'],'fppi':at50['fppi']}
        rows.append(row)
    macro={'model':model,'model_label':MODELS[model],'seed':seed,'fold':fold,'threshold':threshold,
           'n_images':100,'n_gt':sum(r['n_gt'] for r in rows),'tp':sum(r['tp'] for r in rows),
           'fp':sum(r['fp'] for r in rows),'fn':sum(r['fn'] for r in rows),
           **{k:float(np.nanmean([r[k] for r in rows])) for k in FIELDS}}
    macro['fppi']=sum(r['fp'] for r in rows)/100
    tr={'model':model,'seed':seed,'fold':fold,'threshold':threshold,'validation_macro_f1':val_f1}
    return rows,macro,tr,sorted(expected_test)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();root=a.root.resolve();results=root/'results';results.mkdir(exist_ok=True)
    tasks=[(str(root),model,seed,fold) for model in MODELS for seed in SEEDS for fold in range(5)]
    missing=[f'{model}/fold_{fold}_seed_{seed}' for _,model,seed,fold in tasks if not(root/'runs'/model/f'fold_{fold}_seed_{seed}'/'completed.json').is_file()]
    if missing:raise RuntimeError(f'{len(missing)} training jobs remain incomplete: {missing[:8]}')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:out=list(pool.map(evaluate_job,tasks))
    classrows=[r for item in out for r in item[0]];foldrows=[item[1] for item in out];thresholds=[item[2] for item in out]
    expected_images={p.name for p in (root/'dataset/images').glob('*.png')}
    for model in MODELS:
        for seed in SEEDS:
            names=[n for task,result in zip(tasks,out) if task[1:3]==(model,seed) for n in result[3]]
            assert Counter(names)==Counter({n:1 for n in expected_images})
    seedrows=[];seedclass=[];summary=[];summaryclass=[]
    for model in MODELS:
        for seed in SEEDS:
            subset=[r for r in foldrows if r['model']==model and r['seed']==seed];assert len(subset)==5
            seedrows.append({'model':model,'model_label':MODELS[model],'seed':seed,**{k:float(np.nanmean([r[k] for r in subset])) for k in FIELDS}})
            for c,name in enumerate(m.CLASS_NAMES):
                subset=[r for r in classrows if r['model']==model and r['seed']==seed and r['class_id']==c];assert len(subset)==5
                seedclass.append({'model':model,'model_label':MODELS[model],'seed':seed,'class_id':c,'class':name,
                    **{k:float(np.nanmean([r[k] for r in subset])) for k in FIELDS}})
        vals=[r for r in seedrows if r['model']==model]
        summary.append({'model':model,'model_label':MODELS[model],'n_folds':5,'n_seeds':3,
            **{f'{k}_{stat}':float(np.nanmean([r[k] for r in vals]) if stat=='mean' else np.nanstd([r[k] for r in vals],ddof=1)) for k in FIELDS for stat in ['mean','std']}})
        for c,name in enumerate(m.CLASS_NAMES):
            vals=[r for r in seedclass if r['model']==model and r['class_id']==c]
            summaryclass.append({'model':model,'model_label':MODELS[model],'class_id':c,'class':name,
                **{f'{k}_{stat}':float(np.nanmean([r[k] for r in vals]) if stat=='mean' else np.nanstd([r[k] for r in vals],ddof=1)) for k in FIELDS for stat in ['mean','std']}})
    for name,rows in [('per_fold_per_class_metrics',classrows),('per_fold_macro_metrics',foldrows),('fold_thresholds',thresholds),
                      ('per_seed_macro_metrics',seedrows),('per_seed_per_class_metrics',seedclass),
                      ('aggregate_macro_metrics',summary),('aggregate_per_class_metrics',summaryclass)]:csv_write(results/f'{name}.csv',rows)
    metadata={'release':'2026-11-12','images':500,'ground_truth_instances':3000,'class_names':m.CLASS_NAMES,
        'folds':5,'seeds':SEEDS,'train_val_test_per_fold':[300,100,100],'input_size':[640,640],
        'threshold_selection':'Per-fold validation macro-F1; grid 0.01 to 0.99; ties favor higher threshold.',
        'matching':'Class-aware score-ordered greedy one-to-one matching; IoU 0.50 for operating metrics.',
        'prediction_normalization':'Clip xyxy to original 2048x2048 image bounds, remove degenerate boxes, score >= 0.001, at most 100 highest-score predictions per image for all models.',
        'ap':'101-point interpolated precision; AP50-95 averages IoU thresholds 0.50:0.05:0.95.',
        'aggregation':'Class-macro metric per fold, equal mean over five folds per seed, then mean and sample SD over three seeds.',
        'fppi':'Total false positives across all six classes divided by test images; not averaged over classes.',
        'false_alarm':'Per-class FP/(TP+FP); undefined if no detections. Macro averages exclude undefined values.',
        'coverage':'Each image has one outer-test prediction per seed.',
        'external_evaluation':'Not performed in this experiment.'}
    (results/'evaluation_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
