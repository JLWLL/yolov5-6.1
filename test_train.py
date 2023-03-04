import datetime

from train import run
import torch


def do_work(**kwargs):
    save_dir = run(data=kwargs['data'], weights=kwargs['weights'], workers=kwargs['workers'], cfg=kwargs['cfg'],
                   epochs=kwargs['epochs'])
    return save_dir


if __name__ == '__main__':
    model = torch.hub.load(r'E:\Github-code\yolov5-6.1', r'yolov5s', source='local',
                           pretrained=True)  # or yolov5n - yolov5x6, custom
    img = r'E:\Github-code\yolov5-6.1\data\images\bus.jpg'  # or file, Path, PIL, OpenCV, numpy, list
    results = model(img)
    save_dir_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    results.save(save_dir=f'{save_dir_name}')
    kwargs = {
        'data': 'data/voc_ball.yaml'
        , 'weights': 'yolov5s.pt'
        , 'cfg': 'models/yolov5s_ball.yaml'
        , 'epochs': 1
        , 'workers': 16
    }
    save_model_dir = do_work(**kwargs)
    model = torch.hub.load(r'E:\Github-code\yolov5-6.1', rf'best', source='local',
                           pretrained=True)  # or yolov5n - yolov5x6, custom
    results_ = model(img)
    save_dir_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    results.save(save_dir=f'{save_dir_name}')