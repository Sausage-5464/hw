import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper

# --- 页面基本配置 ---
st.set_page_config(page_title="图像处理", layout="wide")
st.title("🌟图像滤波与频域分析 Web App")
st.markdown("本应用实现了常见空间滤波、局部区域梯度计算与可视化，以及频域傅里叶变换分析。")

# --- 左侧边栏：上传图片与任务导航 ---
st.sidebar.header("控制面板")
uploaded_file = st.sidebar.file_uploader("请上传一张图像", type=['png', 'jpg', 'jpeg'])
task = st.sidebar.selectbox(
    "选择要演示的任务", 
    ["1. 空间图像滤波", "2. 图像局部梯度方向", "3. 频域分析与变换"]
)

if uploaded_file is not None:
    # 统一读取为 PIL Image 和 OpenCV 格式，方便不同库调用
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    st.sidebar.image(pil_image, caption="上传的原图", use_container_width=True)

    # ==========================================
    # 任务 1：空间图像滤波
    # ==========================================
    if task == "1. 空间图像滤波":
        st.header("🎛️ 空间图像滤波器对比")
        
        col_ctrl, col_empty = st.columns([1, 2])
        with col_ctrl:
            filter_type = st.selectbox("选择滤波器", ["Box (均值)", "Gaussian (高斯)", "Median (中值)", "Sobel (边缘检测)"])
            kernel_size = st.slider("Kernel Size (卷积核大小，必须为奇数)", 3, 31, 5, step=2)
        
        # 处理图像
        processed_img = image_np.copy()
        if filter_type == "Box (均值)":
            processed_img = cv2.blur(image_np, (kernel_size, kernel_size))
        elif filter_type == "Gaussian (高斯)":
            processed_img = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
        elif filter_type == "Median (中值)":
            processed_img = cv2.medianBlur(image_np, kernel_size)
        elif filter_type == "Sobel (边缘检测)":
            # 计算 X 和 Y 方向的梯度，然后合并幅值
            grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
            grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            sobel_combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            processed_img = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2RGB)

        # 结果展示
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_np, caption="原图", use_container_width=True)
        with col2:
            st.image(processed_img, caption=f"{filter_type} 处理后 (Kernel: {kernel_size}x{kernel_size})", use_container_width=True)


    # ==========================================
    # 任务 2：图像局部梯度方向 (鼠标拖拽框选)
    # ==========================================
    elif task == "2. 图像局部梯度方向":
        st.header("🎯 图像局部区域梯度演示")
        st.write("请在下方图像中 **用鼠标拖拽框选** 感兴趣的局部区域（ROI），右侧将实时计算该区域的梯度方向。")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. 框选原图区域")
            # 使用 streamlit-cropper 进行交互式裁剪
            cropped_pil = st_cropper(
                pil_image, 
                realtime_update=True, 
                box_color='#FF0000',
                aspect_ratio=None # 允许任意比例框选
            )
        
        with col2:
            st.subheader("2. 局部梯度方向可视化")
            if cropped_pil:
                # 将裁剪出来的 PIL 图像转为灰度 numpy 数组
                crop_gray = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2GRAY)
                
                # 计算 Sobel 梯度
                gx = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
                
                # 绘制梯度矢量图 (Quiver Plot)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(crop_gray, cmap='gray')
                
                # 为了防止箭头过于密集，根据图像大小动态设置采样步长
                h, w = crop_gray.shape
                step = max(1, min(h, w) // 25) 
                
                Y, X = np.mgrid[0:h:step, 0:w:step]
                # 提取对应步长位置的梯度
                U = gx[0:h:step, 0:w:step]
                V = gy[0:h:step, 0:w:step]
                
                # matplotlib 中 y 轴向下，所以向量的 y 方向需要取反以匹配视觉
                ax.quiver(X, Y, U, -V, color='red', angles='xy', scale_units='xy', width=0.005)
                ax.set_title("Gradient Directions (Red Arrows)")
                ax.axis('off')
                
                st.pyplot(fig)


    # ==========================================
    # 任务 3：频域分析与变换
    # ==========================================
    elif task == "3. 频域分析与变换":
        st.header("🌊 频域图像滤波与几何变换对频谱的影响")
        
        col_ctrl, col_empty = st.columns([1, 2])
        with col_ctrl:
            transform_type = st.radio("对原图进行几何变换:", ["无变换", "旋转 (Rotation)", "缩放 (Scaling)"])
            
            # 动态显示控制滑块
            if transform_type == "旋转 (Rotation)":
                angle = st.slider("旋转角度", -180, 180, 45)
            elif transform_type == "缩放 (Scaling)":
                scale = st.slider("缩放比例", 0.1, 3.0, 1.5, step=0.1)

        # 1. 执行几何变换
        h, w = image_gray.shape
        transformed_gray = image_gray.copy()
        
        if transform_type == "旋转 (Rotation)":
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            transformed_gray = cv2.warpAffine(image_gray, M, (w, h))
        elif transform_type == "缩放 (Scaling)":
            transformed_gray = cv2.resize(image_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # 2. 傅里叶变换与频谱计算
        # 使用 numpy 的 FFT
        f = np.fft.fft2(transformed_gray)
        # 将低频部分移动到图像中心
        fshift = np.fft.fftshift(f)
        # 计算频谱幅值并进行对数变换以便于观察
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        # 3. 结果展示
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(transformed_gray, caption="当前空域图像 (Gray)", use_container_width=True, clamp=True)
        with col2:
            # 使用 matplotlib 绘制频谱图，映射颜色更清晰
            fig, ax = plt.subplots()
            cax = ax.imshow(magnitude_spectrum, cmap='magma')
            ax.set_title("Fourier Magnitude Spectrum")
            ax.axis('off')
            fig.colorbar(cax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            
        st.info("💡 **观察提示**：\n"
                "- **旋转**：空域图像旋转时，频域的频谱图也会发生**相同角度的旋转**。\n"
                "- **缩放**：空域图像放大时，频域的频谱图会**缩小**（高频向中心收缩）；反之亦然。")

else:
    st.info("👋 欢迎！请先在左侧边栏上传一张测试图片开始您的实验。")