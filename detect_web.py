import torch
import datetime
# 模型
model = torch.hub.load(r'C:\Users\Administrator\Desktop\JLWLbot\yolov5-6.1', r'yolov5s6',source='local', pretrained=True)  # or yolov5n - yolov5x6, custom

# 图像
img = r'C:\Users\Administrator\Desktop\JLWLbot\data\images\4\exp2\110.jpeg'  # or file, Path, PIL, OpenCV, numpy, list

# 推理
results = model(img)

# 结果
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
save_dir_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
results.save(save_dir=f'C:\\Users\\Administrator\\Desktop\\JLWLbot\\data\\images\\4\\{save_dir_name}')