"""Regenerate validation/test predictions from an existing fivefold checkpoint."""
from pathlib import Path
import argparse, json, os

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',required=True,choices=['yolo11m','yolov10m','rtdetr_l','fasterrcnn_r50_fpn_v2','retinanet_r50_fpn_v2','fcos_r50_fpn'])
    p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--device',default='0',help='Physical GPU index, or cpu.')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,help='Default: predictions_<split>.json in the run directory.')
    p.add_argument('--limit',type=int,help='Smoke-test only; requires an explicit output filename.')
    a=p.parse_args()
    if a.limit is not None and (a.output is None or a.limit<1):p.error('--limit requires --output and a positive count')
    os.environ['CUDA_VISIBLE_DEVICES']='' if a.device=='cpu' else a.device
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from torch.utils.data import DataLoader, Subset
    from train_cv import DetectionDataset, collate, predict_fasterrcnn, load_paths, serialize_ultralytics_predictions
    torch.set_num_threads(4)
    root=a.root.resolve();run=root/'runs'/a.model/f'fold_{a.fold}_seed_{a.seed}'
    ultra=a.model in {'yolo11m','yolov10m','rtdetr_l'}
    checkpoint=a.checkpoint or (run/'weights/best.pt' if ultra else run/'best.pt')
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    paths=load_paths(root/'splits'/f'fold_{a.fold}'/f'{a.split}.txt')
    if a.limit is not None:paths=paths[:a.limit]
    if ultra:
        from ultralytics import YOLO, RTDETR
        model=(RTDETR if a.model=='rtdetr_l' else YOLO)(str(checkpoint))
        results=model.predict(source=[str(x) for x in paths],imgsz=640,conf=0.001,iou=0.70,max_det=100,
                              device='cpu' if a.device=='cpu' else 0,stream=True,verbose=False)
        images=serialize_ultralytics_predictions(results,paths)
    else:
        from model_loader import build_torchvision_model
        device=torch.device('cpu' if a.device=='cpu' else 'cuda:0')
        model=build_torchvision_model(a.model,checkpoint,device)
        dataset=DetectionDataset(root/'splits'/f'fold_{a.fold}'/f'{a.split}.txt')
        if a.limit is not None:dataset=Subset(dataset,range(len(paths)))
        loader=DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0,collate_fn=collate)
        images=predict_fasterrcnn(model,loader,device)
    output=a.output or run/f'predictions_{a.split}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    payload={'model':a.model,'fold':a.fold,'seed':a.seed,'split':a.split,'image_size':640,
             'checkpoint':str(checkpoint),'images':images}
    output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(f'Saved {len(images)} images to {output}')

if __name__=='__main__':main()
