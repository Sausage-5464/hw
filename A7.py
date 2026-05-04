import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTMAEForPreTraining

# --- 1. 页面与环境配置 ---
st.set_page_config(page_title="真实 MAE 自监督学习演示", layout="wide")
st.title("🔥 真实自监督学习模型：MAE 图像遮挡重建")
st.markdown("""
本应用**不使用模拟数据**。后端直接加载了真实的 `facebook/vit-mae-base` 预训练大模型。
它将对你上传的图片进行 Patch 级别的随机遮挡，并利用真实的 Transformer 解码器进行像素级重建。
你可以直观感受到大型自监督模型在缺失大量信息时的“脑补”能力。
""")

# --- 2. 加载真实的预训练模型 ---
@st.cache_resource(show_spinner=False)
def load_real_model():
    # 使用 transformers 加载真实的 ViT-MAE 模型和图像处理器
    processor = ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
    model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
    model.eval() # 设置为推理模式
    return processor, model

with st.spinner("📦 正在加载真实预训练模型 facebook/vit-mae-base (初次运行会自动下载权重，请稍候)..."):
    processor, model = load_real_model()

# --- 3. 图像后处理与可视化核心逻辑 ---
def denormalize(tensor):
    """撤销 ImageNet 归一化，以便正确可视化"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def get_real_reconstruction(pixel_values, outputs, patch_size=16):
    """解析真实模型的输出，生成遮挡图和重建图"""
    # 1. 提取预测的像素值 (1, 196, 768) -> (1, 3, 224, 224)
    pred = outputs.logits 
    h = w = 224 // patch_size
    pred_imgs = pred.reshape(1, h, w, patch_size, patch_size, 3)
    pred_imgs = torch.einsum('nhwpqc->nchpwq', pred_imgs)
    pred_imgs = pred_imgs.reshape(1, 3, h * patch_size, h * patch_size)
    
    # 2. 提取并还原真实的 Mask (1: masked, 0: unmasked)
    mask = outputs.mask 
    mask_imgs = mask.reshape(1, h, w, 1, 1, 1).expand(-1, -1, -1, patch_size, patch_size, 3)
    mask_imgs = torch.einsum('nhwpqc->nchpwq', mask_imgs)
    mask_imgs = mask_imgs.reshape(1, 3, h * patch_size, h * patch_size)
    
    # 3. 生成展示用的图像
    # 被遮挡的输入图 (将 masked 部分设为灰色 0.5)
    masked_input = pixel_values.clone()
    masked_input[mask_imgs == 1] = 0.0 # 设为黑色遮挡
    
    # 真实的模型重建结果 (未遮挡部分保留原图，遮挡部分使用模型预测)
    reconstruction = pixel_values * (1 - mask_imgs) + pred_imgs * mask_imgs
    
    return masked_input[0], reconstruction[0]

def tensor_to_image(tensor):
    """将 Tensor 转换为可供 Streamlit 显示的 Numpy 数组"""
    return denormalize(tensor).permute(1, 2, 0).detach().numpy()

# --- 4. 侧边栏交互 ---
st.sidebar.header("⚙️ 真实推理参数设置")
uploaded_file = st.sidebar.file_uploader("上传一张真实图片", type=["jpg", "png", "jpeg"])
st.sidebar.markdown("---")
st.sidebar.write("MAE 默认遮挡率为 75%。我们对比两种极端情况：")
mask_ratio_1 = st.sidebar.slider("对比组 A: 较低的遮挡比例", 0.1, 0.9, 0.40, step=0.05)
mask_ratio_2 = st.sidebar.slider("对比组 B: 极高的遮挡比例", 0.1, 0.9, 0.85, step=0.05)

# --- 5. 执行真实前向传播 ---
if uploaded_file:
    # 加载图片
    image = Image.open(uploaded_file).convert("RGB")
    
    # 使用真实的 Processor 进行 ImageNet 标准预处理 (Resize 224x224, 归一化等)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values

    # 显示原图
    st.subheader("🖼️ 原始图片 (Resize to 224x224)")
    st.image(tensor_to_image(pixel_values[0]), width=224)
    st.divider()

    st.subheader("🧠 真实模型推理结果对比")
    col1, col2 = st.columns(2)

    with torch.no_grad():
        # --- 对比组 A 的真实推理 ---
        model.config.mask_ratio = mask_ratio_1 # 动态修改真实模型的 Mask Ratio
        outputs_1 = model(pixel_values)
        loss_1 = outputs_1.loss.item() # 获取真实的重建 Loss
        masked_img_1, recon_img_1 = get_real_reconstruction(pixel_values, outputs_1)

        # --- 对比组 B 的真实推理 ---
        model.config.mask_ratio = mask_ratio_2
        outputs_2 = model(pixel_values)
        loss_2 = outputs_2.loss.item() # 获取真实的重建 Loss
        masked_img_2, recon_img_2 = get_real_reconstruction(pixel_values, outputs_2)

    # --- 渲染对比组 A ---
    with col1:
        st.info(f"设置 A (遮挡率: {mask_ratio_1*100:.0f}%)")
        st.image(tensor_to_image(masked_img_1), caption="送入模型的 Tensor (黑色为真实被 Mask 的 Patch)")
        st.image(tensor_to_image(recon_img_1), caption="MAE 真实的像素重建结果")
        st.metric("真实重建 Loss (MSE on masked patches)", f"{loss_1:.4f}")

    # --- 渲染对比组 B ---
    with col2:
        st.info(f"设置 B (遮挡率: {mask_ratio_2*100:.0f}%)")
        st.image(tensor_to_image(masked_img_2), caption="送入模型的 Tensor (高遮挡难度急剧上升)")
        st.image(tensor_to_image(recon_img_2), caption="MAE 真实的像素重建结果")
        st.metric("真实重建 Loss (MSE on masked patches)", f"{loss_2:.4f}", delta=f"{loss_2 - loss_1:.4f}", delta_color="inverse")
        st.caption("注：遮挡比例越高，需要猜测的像素越多，Loss 通常越大，但你可以观察到模型仍能把握大致的轮廓和颜色。")

else:
    st.warning("👈 请在侧边栏上传一张图片以启动真实模型推理。")