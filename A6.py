import streamlit as st
import torch
import torchvision
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
import time

# Set page config
st.set_page_config(page_title="Vibe Coding: Computer Vision Demo", layout="wide")

st.title("计算机视觉多任务演示系统")
st.markdown("本应用集成了 FCN 语义分割、Faster R-CNN 目标检测和 Mask R-CNN 实例分割。")

# --- Model Loading Functions ---
@st.cache_resource
def load_fcn():
    # Load FCN ResNet50
    model = models.segmentation.fcn_resnet50(weights='DEFAULT').eval()
    return model

@st.cache_resource
def load_faster_rcnn():
    # Load Faster R-CNN ResNet50 FPN
    model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT').eval()
    return model

@st.cache_resource
def load_mask_rcnn():
    # Load Mask R-CNN ResNet50 FPN
    model = models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT').eval()
    return model

# COCO Class Names for Detection/Segmentation
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'clothing', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- Sidebar Controls ---
st.sidebar.header("配置中心")
task = st.sidebar.selectbox("选择任务", ["FCN 语义分割", "Faster R-CNN 目标检测", "Mask R-CNN 实例分割", "性能对比"])
confidence_threshold = st.sidebar.slider("置信度阈值", 0.1, 1.0, 0.5)

uploaded_file = st.sidebar.file_uploader("上传一张图片...", type=["jpg", "jpeg", "png"])

def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=pottedplant, 17=sheep, 18=sofa, 19=train, 20=tvmonitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# --- Execution ---

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess(img)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="原始图片", use_container_width=True)

    start_time = time.time()

    if task == "FCN 语义分割":
        model = load_fcn()
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        seg_map = torch.argmax(output, dim=0).detach().cpu().numpy()
        rgb_map = decode_segmap(seg_map)
        inference_time = time.time() - start_time
        
        with col2:
            st.image(rgb_map, caption=f"分割结果 (用时: {inference_time:.2f}s)", use_container_width=True)

    elif task == "Faster R-CNN 目标检测":
        model = load_faster_rcnn()
        with torch.no_grad():
            prediction = model(input_tensor)[0]
        
        draw = ImageDraw.Draw(img)
        for score, label, box in zip(prediction['scores'], prediction['labels'], prediction['boxes']):
            if score > confidence_threshold:
                box = box.cpu().numpy()
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
                draw.text((box[0], box[1]), f"{COCO_CLASSES[label]}: {score:.2f}", fill="red")
        
        inference_time = time.time() - start_time
        with col2:
            st.image(img, caption=f"检测结果 (用时: {inference_time:.2f}s)", use_container_width=True)

    elif task == "Mask R-CNN 实例分割":
        model = load_mask_rcnn()
        with torch.no_grad():
            prediction = model(input_tensor)[0]
        
        # Convert to numpy for OpenCV drawing
        img_np = np.array(img)
        masks = prediction['masks'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        
        for i in range(len(masks)):
            if scores[i] > confidence_threshold:
                mask = masks[i, 0]
                color = np.random.randint(0, 255, (3,)).tolist()
                # Apply mask overlay
                img_np[mask > 0.5] = img_np[mask > 0.5] * 0.5 + np.array(color) * 0.5
                # Draw bounding box
                box = prediction['boxes'][i].cpu().numpy().astype(int)
                cv2.rectangle(img_np, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(img_np, COCO_CLASSES[labels[i]], (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        inference_time = time.time() - start_time
        with col2:
            st.image(img_np, caption=f"实例分割结果 (用时: {inference_time:.2f}s)", use_container_width=True)

    elif task == "性能对比":
        st.subheader("不同模型推理性能对比")
        results = []
        models_to_test = {
            "FCN (ResNet50)": load_fcn,
            "Faster R-CNN (ResNet50)": load_faster_rcnn,
            "Mask R-CNN (ResNet50)": load_mask_rcnn
        }
        
        progress_bar = st.progress(0)
        for i, (name, loader) in enumerate(models_to_test.items()):
            m = loader()
            t0 = time.time()
            with torch.no_grad():
                _ = m(input_tensor)
            dt = time.time() - t0
            results.append({"模型": name, "推理时间 (s)": round(dt, 4)})
            progress_bar.progress((i + 1) / len(models_to_test))
            
        st.table(results)
        st.bar_chart({r["模型"]: r["推理时间 (s)"] for r in results})

else:
    st.info("请在左侧边栏上传一张图片以开始。")

