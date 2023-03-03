import torch
import datetime
# 模型
model = torch.hub.load(r'E:\Github-code\yolov5-6.1', r'yolov5s',source='local', pretrained=True)  # or yolov5n - yolov5x6, custom

# 图像
img = r'E:\Github-code\yolov5-6.1\data\images\bus.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# 推理
results = model(img)

# 结果
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
save_dir_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
results.save(save_dir=f'{save_dir_name}')