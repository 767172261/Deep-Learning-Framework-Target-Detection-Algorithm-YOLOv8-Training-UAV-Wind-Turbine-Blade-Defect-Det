深度学习框架目标检测算法YOLOv8训练无人机风力发电叶片缺陷检测数据集风电叶片缺陷检测数据集的训练及应用 推理识别检测鸟类、航空燃油（油渍）、污迹、附着的沙子和昆虫等缺陷
@[toc]
以下文字及代码仅供参考学习使用。

**数据集描述：
风电叶片缺陷检测数据集，共1300余张超高分辨率各种污染物包括：鸟类、航空燃油（油渍）、污迹、附着的沙子和昆虫。 叶片缺陷数据集 风力发电机叶片缺陷数据集 yolo格式 23 类** 

---

### 📋 风电叶片缺陷检测数据集类别中英文对照表

| 序号 | 英文名称                                      | 中文翻译                                 |
|------|-----------------------------------------------|------------------------------------------|
| 1    | Surface-contamination-dirt                    | 表面污染-污垢                             |
| 2    | add-on-LPS-OK                                 | 附加件-LPS（定位系统）-正常               |
| 3    | add-on-LPS-other                              | 附加件-LPS（定位系统）-其他               |
| 4    | add-on-LPS-worn or burnt                      | 附加件-LPS（定位系统）-磨损或烧损         |
| 5    | add-on-VG-OK                                  | 附加件-VG（涡流发生器）-正常              |
| 6    | add-on-drainhole-OK                           | 附加件-排水孔-正常                        |
| 7    | add-on-dynotail-OK                            | 附加件-动力尾翼-正常                      |
| 8    | add-on-markings-CoG                           | 附加件-标记-重心（CoG）                   |
| 9    | add-on-markings-lifting point                 | 附加件-标记-吊点                          |
| 10   | add-on-markings-text                          | 附加件-标记-文字                          |
| 11   | add-on-serration-OK                           | 附加件-锯齿结构-正常                      |
| 12   | add-on-tip-OK                                 | 附加件-叶尖-正常                          |
| 13   | leading edge-erosion-coating or LEP only      | 前缘侵蚀-仅涂层或LEP（前缘保护）          |
| 14   | leading edge-erosion-continuous or deep       | 前缘侵蚀-连续或深层                       |
| 15   | leading edge-erosion-eroded tip               | 前缘侵蚀-侵蚀叶尖                         |
| 16   | leading edge-erosion-spotty or laminate       | 前缘侵蚀-斑点状或层压板                   |
| 17   | surface-burn damage-deep                      | 表面烧伤损伤-深层                          |
| 18   | surface-burn damage-superficial               | 表面烧伤损伤-表层                          |
| 19   | surface-coating damage-superficial            | 表面涂层损伤-表层                          |
| 20   | surface-contamination-other                   | 表面污染-其他                             |
| 21   | surface-crack or laminate defect-superficial  | 表面裂纹或层压缺陷-表层                   |
| 22   | trailing edge-crack-deep                      | 后缘裂纹-深层                              |
| 23   | trailing edge-crack-superficial               | 后缘裂纹-表层                              |

---

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/89bca9c6533a4dee8ee39f15545e617f.png)
配置YAML

```bash
train: ../train/images val: ../valid/images test: ../test/images nc: 23 names: ['Surface-contamination-dirt', 'add-on-LPS-OK', 'add-on-LPS-other', 'add-on-LPS-worn or burnt', 'add-on-VG-OK', 'add-on-drainhole-OK', 'add-on-dynotail-OK', 'add-on-markings-CoG', 'add-on-markings-lifting point', 'add-on-markings-text', 'add-on-serration-OK', 'add-on-tip-OK', 'leading edge-erosion-coating or LEP only', 'leading edge-erosion-continuous or deep', 'leading edge-erosion-eroded tip', 'leading edge-erosion-spotty or laminate', 'surface-burn damage-deep', 'surface-burn damage-superficial', 'surface-coating damage-superficial', 'surface-contamination-other', 'surface-crack or laminate defect-superficial', 'trailing edge-crack-deep', 'trailing edge-crack-superficial']
```
基于YOLOv8的风电叶片缺陷检测系统，从环境配置、数据准备、模型训练、推理实现等方面进行详细说明。以下是完整的步骤指南。仅供参考学习，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/47c32d867ec64ca99121b524438b5b4b.png)

### 一、环境配置

#### 1. 安装CUDA驱动
确保同学你的计算机上安装了与你的NVIDIA GPU兼容的CUDA版本。你可以从[NVIDIA官网](https://developer.nvidia.com/cuda-downloads)下载并安装适合你操作系统的CUDA Toolkit。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/843999c92a0240d9a8852bdc99dd8dfb.png)

#### 2. 安装Anaconda
访问[Anaconda官网](https://www.anaconda.com/products/distribution)，根据你的操作系统下载并安装Anaconda。

#### 3. 创建Python虚拟环境
打开终端或命令提示符，创建一个新的Anaconda环境：

```bash
conda create -n wind_turbine_defects python=3.9
conda activate wind_turbine_defects
```

#### 4. 安装必要的依赖项
在激活的环境中安装必要的库：

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib numpy
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/26ccddb7d18a49aba9ec4431d2bc6a12.png)

### 二、数据准备

假设同学你的数据集已经按照以下结构组织，并且每个图像都有一个对应的YOLO格式的标签文件（`.txt`）：

```
wind_turbine_defects_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

并且你需要创建一个描述数据集路径和类别的`.yaml`文件（例如`data.yaml`）：

```yaml
train: ./wind_turbine_defects_dataset/images/train
val: ./wind_turbine_defects_dataset/images/val

nc: 23  # 类别数量
names: ['Surface-contamination-dirt', 'add-on-LPS-OK', 'add-on-LPS-other', 'add-on-LPS-worn or burnt', 'add-on-VG-OK', 'add-on-drainhole-OK', 'add-on-dynotail-OK', 'add-on-markings-CoG', 'add-on-markings-lifting point', 'add-on-markings-text', 'add-on-serration-OK', 'add-on-tip-OK', 'leading edge-erosion-coating or LEP only', 'leading edge-erosion-continuous or deep', 'leading edge-erosion-eroded tip', 'leading edge-erosion-spotty or laminate', 'surface-burn damage-deep', 'surface-burn damage-superficial', 'surface-coating damage-superficial', 'surface-contamination-other', 'surface-crack or laminate defect-superficial', 'trailing edge-crack-deep', 'trailing edge-crack-superficial']  # 类别名称
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/81bec82dd0df4c7792ee11056db84cc9.png)

### 三、使用YOLOv8进行训练、推理

#### 训练代码

首先，下载官方预训练模型（YOLOv8s）：

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

然后，开始训练：

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8s.pt')

# 开始训练
results = model.train(data='path/to/data.yaml', epochs=100, imgsz=640)
```

或者直接通过命令行执行：

```bash
yolo train data=path/to/data.yaml model=yolov8s.pt epochs=100 imgsz=640 batch=16 device=0
```

#### 推理代码

加载训练好的模型进行预测：

```python
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

def detect_wind_turbine_defects(image_path, model):
    # 使用模型进行预测
    results = model.predict(image_path, conf=0.25)  # 调整conf阈值以适应你的需求
    
    # 可视化结果
    for result in results:
        boxes = result.boxes.numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            score = box[4]
            label = result.names[int(box[5])]
            cv2.rectangle(result.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result.orig_img, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return result.orig_img

# 示例调用
result_image = detect_wind_turbine_defects('path_to_your_test_image.jpg', model)
cv2.imshow('Wind Turbine Defect Detection Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 四、评估模型性能

在验证集上评估模型性能：

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

# 在验证集上评估模型性能
metrics = model.val(data='path/to/data.yaml')

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
```

根据你训练好的 **YOLOv8 模型权重**（如 `best.pt`），你可以构建一个完整的 **深度学习风电叶片缺陷检测系统**，用于图像、视频或实时摄像头的缺陷识别。以下是一个从 **环境准备 → 加载模型 → 图像/视频推理 → 结果可视化 → 部署建议** 的完整指南。

---

## ✅ 一、环境准备

确保你的环境中已安装了所有必要的依赖项：

```bash
conda create -n turbine_defect python=3.9
conda activate turbine_defect

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib numpy
```

> ✅ 确保你使用的是 GPU 支持版本的 PyTorch，并且 CUDA 驱动已正确安装。

---

## ✅ 二、加载训练好的 YOLOv8 模型权重

假设你已经训练好了模型，并得到了最佳权重文件：`best.pt`，位于路径：`runs/detect/train/weights/best.pt`

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')
```

---

## ✅ 三、图像推理与结果可视化

### 📷 单张图像推理函数

```python
import cv2

def detect_turbine_defects(image_path, model):
    # 使用模型进行预测
    results = model.predict(source=image_path, conf=0.25)  # 可调整置信度阈值
    
    # 获取原始图像并绘制检测框
    for result in results:
        annotated_img = result.plot()  # 自动绘制边界框和标签
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)  # 转换颜色空间
        return annotated_img
```

### 🖼️ 示例调用并显示图像

```python
result_image = detect_turbine_defects('test_images/turbine_001.jpg', model)
cv2.imshow('Wind Turbine Defect Detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## ✅ 四、视频流推理（支持摄像头 / 视频文件）

### 🎥 实时摄像头推理

```python
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_frame = detect_turbine_defects(frame, model)
    cv2.imshow('Real-time Wind Turbine Defect Detection', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 🎞️ 视频文件推理

```python
video_path = 'videos/test_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_frame = detect_turbine_defects(frame, model)
    cv2.imshow('Video Wind Turbine Defect Detection', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ✅ 五、保存推理结果到文件

### 💾 保存单张图像结果

```python
output_image = detect_turbine_defects('test_images/turbine_001.jpg', model)
cv2.imwrite('outputs/turbine_001_result.jpg', output_image)
```

### 🎥 保存视频推理结果

```python
out = cv2.VideoWriter('outputs/output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))

cap = cv2.VideoCapture('videos/test_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_frame = detect_turbine_defects(frame, model)
    out.write(result_frame)

    cv2.imshow('Saving Result...', result_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

---

## ✅ 六、部署方案建议

### 1. Web API 服务（Flask + Ultralytics）

#### 安装 Flask：

```bash
pip install flask
```

#### 创建 `app.py` 接口示例：

```python
from flask import Flask, request, send_file
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
model = YOLO('runs/detect/train/weights/best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    image_path = 'temp.jpg'
    file.save(image_path)

    result_img = detect_turbine_defects(image_path, model)
    cv2.imwrite('output.jpg', result_img)

    return send_file('output.jpg', mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

启动服务后，可以通过 POST 请求上传图像进行检测。

---

### 2. 边缘设备部署（Jetson Nano / Orin）

- 使用 TensorRT 转换 `.pt` 模型为 `.engine` 文件，提升推理速度。
- 使用 DeepStream SDK 处理多路视频流。

### 3. Docker 容器化部署

创建一个 `Dockerfile` 来打包整个应用及其依赖：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

构建并运行容器：

```bash
docker build -t turbine-defect-api .
docker run -p 5000:5000 turbine-defect-api
```

---

以上文字及代码仅供参考学习使用。
