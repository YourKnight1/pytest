# pytest
test


import cv2
from ultralytics import YOLO

model = YOLO('yolov5s.pt')

model.conf = 0.25
model.iou  = 0.45
"""
# 核心推理参数
# source (str, 默认 'ultralytics/assets')：输入数据源，支持本地图片/视频路径、目录、URL、摄像头设备 ID 等多种类型。
# conf (float, 默认 0.25)：置信度阈值，低于该值的检测结果会被过滤。
# iou (float, 默认 0.7)：NMS 的 IoU（Intersection over Union）阈值，用于去除重叠框。
# imgsz (int 或 tuple, 默认 640)：推理时的输入图像尺寸，可为单一整数（方形缩放）或 (height, width)。
# half (bool, 默认 False)：是否启用半精度（FP16）推理，以加速 GPU 上的推理。
# device (str, 默认 None)：指定运行设备，如 "cpu"、"cuda:0" 或 GPU 索引 "0"。
# batch (int, 默认 1)：批量大小，仅对目录、视频文件或 .txt 列表源有效。
# max_det (int, 默认 300)：每张图像最大检测数量，防止密集场景输出过多框。
# vid_stride (int, 默认 1)：视频输入的帧跳步，1 表示处理所有帧，>1 可加速处理但会跳帧。
# stream_buffer (bool, 默认 False)：视频流模式下是否缓存帧，False 会丢弃旧帧，True 会排队但可能造成延迟。
# visualize (bool, 默认 False)：开启可视化中间特征图，用于调试与模型解释。
# augment (bool, 默认 False)：测试时启用数据增强（TTA），可提升准确率但降低速度。
# agnostic_nms (bool, 默认 False)：类别无关 NMS，不同类别也可相互抑制。
# classes (list[int], 默认 None)：仅返回指定类别 ID 的检测结果。
# retina_masks (bool, 默认 False)：输出原图分辨率的高精度分割掩码。
# embed (list[int], 默认 None)：从指定层提取特征向量或嵌入，用于下游任务。
# project (str, 默认 None)：保存目录的根路径，仅当 save=True 时生效。
# name (str, 默认 None)：本次预测的子目录名，配合 project 组织输出。
# stream (bool, 默认 False)：流式模式，返回生成器 (generator) 而非列表，适合长视频或大批量输入。
# verbose (bool, 默认 True)：是否在终端打印详细推理日志。

# 可视化与保存参数
# show (bool, 默认 False)：是否弹窗显示带标注的图像/视频。
# save (bool, 默认 False／True)：是否将带标注结果保存到文件；CLI 默认 True，Python API 默认 False。
# save_frames (bool, 默认 False)：视频输入时是否逐帧另存为图片。
# save_txt (bool, 默认 False)：以文本格式保存检测框与置信度，方便后续处理。
# save_conf (bool, 默认 False)：在文本文件中附带置信度信息。
# save_crop (bool, 默认 False)：将检测到的目标裁剪后保存为独立图片。
# show_labels (bool, 默认 True)：可视化时显示类别标签。
# show_conf (bool, 默认 True)：可视化时显示置信度数值。
# show_boxes (bool, 默认 True)：可视化时绘制边界框。
# line_width (int 或 None, 默认 None)：边界框线宽；None 时自动根据图像尺寸调整。
# font_size (float 或 None, 默认 None)：注释文字大小；None 时自动缩放。
# font (str, 默认 'Arial.ttf')：注释文字的字体名字或路径。
# pil (bool, 默认 False)：是否以 PIL Image 格式返回结果而非 NumPy 数组。
# kpt_radius (int, 默认 5)：关键点可视化时的点半径。
# kpt_line (bool, 默认 True)：关键点可视化时是否连线。
# masks (bool, 默认 True)：是否在可视化时显示分割掩码。
# probs (bool, 默认 True)：是否显示分类概率。
# filename (str, 默认 None)：当 save=True 时指定保存文件的完整路径。
# color_mode (str, 默认 'class')：可视化配色模式，如 'instance' 或 'class'。
# txt_color (tuple[int,int,int], 默认 (255,255,255))：文本注释的 RGB 颜色。
"""
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 直接使用 high-level predict（内部完成预处理、推理、NMS）
    results = model.predict(source=frame, show=False)  # 返回 [Results]

    # 绘制结果并展示
    for res in results:
        vis = res.plot()  # 返回带框的图像 :contentReference[oaicite:8]{index=8}
        cv2.imshow('YOLOv5 实时检测', vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
