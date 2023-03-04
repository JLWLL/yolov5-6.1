from my_detect import parse_opt, run_detect
from train import run


def do_work(args):
    save_dir = run(data=args['data'], weights=args['weights'], workers=args['workers'], cfg=args['cfg'],
                   epochs=args['epochs'])
    return save_dir


if __name__ == '__main__':
    # 第一次示范推理
    img = r'E:\Github-code\yolov5-6.1\data\images\bus.jpg'  # or file, Path, PIL, OpenCV, numpy, list
    s1 = run_detect(parse_opt('yolov5s.pt', img))  # 权重名称，图片地址
    print(f'第一次结果保存地址{s1}')
    # 训练一个全新的模型
    args_ = {
        'data': 'data/voc_ball.yaml'
        , 'weights': 'yolov5s.pt'
        , 'cfg': 'models/yolov5s_ball.yaml'
        , 'epochs': 2
        , 'workers': 16
    }
    save_model_dir = do_work(args_)
    # 用新训练出的模型的best结果用于下一次推理
    s2 = run_detect(parse_opt(f'{save_model_dir.save_dir}/weights/best.pt', img))
    print(f'第二次结果保存地址{s2}')
