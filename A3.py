import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 页面配置
st.set_page_config(page_title="CV 特征检测与匹配系统", layout="wide")
st.title("图像特征检测与匹配")
st.markdown("---")

# 侧边栏导航
st.sidebar.title("功能导航")
app_mode = st.sidebar.selectbox("选择实验模块",
    ["1. Canny 边缘检测 (NMS对比)", 
     "2. 特征点检测 (Harris vs SIFT)", 
     "3. 图像匹配流程可视化", 
     "4. 多图全景拼接 (Blending对比)"]
)

# 辅助函数：加载图像
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    return np.array(image)

# ==========================================
# 模块 1: Canny 边缘检测及非极大值抑制(NMS)对比
# ==========================================
if app_mode == "1. Canny 边缘检测 (NMS对比)":
    st.header("1. Canny 边缘检测与 NMS 可视化")
    uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img = load_image(uploaded_file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 1. 高斯滤波 & 梯度计算
        blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
        gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 2. 非极大值抑制 (NMS) 手动实现用于可视化
        M, N = mag.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = angle % 180
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    q = 255; r = 255
                    # 0度
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = mag[i, j+1]; r = mag[i, j-1]
                    # 45度
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = mag[i+1, j-1]; r = mag[i-1, j+1]
                    # 90度
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = mag[i+1, j]; r = mag[i-1, j]
                    # 135度
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = mag[i-1, j-1]; r = mag[i+1, j+1]

                    if (mag[i,j] >= q) and (mag[i,j] >= r):
                        Z[i,j] = mag[i,j]
                    else:
                        Z[i,j] = 0
                except IndexError as e:
                    pass
                    
        nms_norm = cv2.normalize(Z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3. 双阈值 & 最终 OpenCV Canny
        final_canny = cv2.Canny(blur, 50, 150)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(mag_norm, caption="NMS 前 (梯度幅值)", use_container_width=True)
        with col2:
            st.image(nms_norm, caption="NMS 后 (非极大值抑制)", use_container_width=True)
        with col3:
            st.image(final_canny, caption="最终 Canny 结果 (包含滞后阈值)", use_container_width=True)

# ==========================================
# 模块 2: 特征点检测 (Harris, SIFT)
# ==========================================
elif app_mode == "2. 特征点检测 (Harris vs SIFT)":
    st.header("2. 特征点检测可视化")
    uploaded_file = st.file_uploader("上传一张图片进行特征检测", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img = load_image(uploaded_file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Harris 角点检测")
            harris_img = img.copy()
            dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
            dst = cv2.dilate(dst, None) # 膨胀以易于可视化
            # 标记角点：画红色圆圈
            harris_img[dst > 0.01 * dst.max()] = [255, 0, 0] 
            st.image(harris_img, caption="Harris 特征点", use_container_width=True)
            
        with col2:
            st.subheader("SIFT 特征检测")
            sift = cv2.SIFT_create()
            kp = sift.detect(gray, None)
            # 绘制 SIFT 特征点（带方向和尺度的圆圈）
            sift_img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            st.image(sift_img, caption="SIFT 特征点 (展示尺度与方向)", use_container_width=True)

# ==========================================
# 模块 3: 图像匹配流程可视化
# ==========================================
elif app_mode == "3. 图像匹配流程可视化":
    st.header("3. 图像匹配与 RANSAC 筛选")
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        img1_file = st.file_uploader("上传图片 1", type=["jpg", "png", "jpeg"])
    with col_up2:
        img2_file = st.file_uploader("上传图片 2", type=["jpg", "png", "jpeg"])
        
    if img1_file and img2_file:
        img1 = load_image(img1_file)
        img2 = load_image(img2_file)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # 1. SIFT 检测与描述
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # 2. 初始匹配 (KNN)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # 应用 Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        img_initial_match = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # 3. RANSAC 剔除误匹配
        if len(good_matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # 计算单应性矩阵并使用 RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            
            # 绘制内点 (Inliers)
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
            img_ransac_match = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
            
            st.subheader("流程对比")
            st.image(img_initial_match, caption=f"初始匹配 (应用 Ratio Test后): {len(good_matches)} 对", use_container_width=True)
            st.image(img_ransac_match, caption=f"RANSAC 筛选后对齐: {sum(matchesMask)} 对", use_container_width=True)
        else:
            st.warning("特征点不足，无法执行 RANSAC。")

# ==========================================
# 模块 4: 多幅图像全景拼接 (Blending 对比)
# ==========================================
elif app_mode == "4. 多图全景拼接 (Blending对比)":
    st.header("4. 全景拼接与 Blending 策略")
    uploaded_files = st.file_uploader("上传多张重叠图片 (建议2-3张)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if len(uploaded_files) >= 2:
        images = [load_image(f) for f in uploaded_files]
        st.write(f"已加载 {len(images)} 张图片。")
        
        if st.button("开始拼接"):
            with st.spinner("正在拼接..."):
                # 策略 1: OpenCV Stitcher (自带高级多频段 Blending 和曝光补偿)
                stitcher = cv2.Stitcher_create()
                status, pano_blended = stitcher.stitch(images)
                
                # 策略 2: 简单的单应性变换 + 暴力覆盖 (Naive 无 Blending) 用于对比
                # 仅展示前两张图的 Naive 拼接
                img1_gray = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(images[1], cv2.COLOR_RGB2GRAY)
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(img1_gray, None)
                kp2, des2 = sift.detectAndCompute(img2_gray, None)
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                
                if len(good) > 4:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    # 粗糙拼接：计算画布大小
                    h1, w1 = images[0].shape[:2]
                    h2, w2 = images[1].shape[:2]
                    pano_naive = cv2.warpPerspective(images[0], H, (w1 + w2, h2))
                    pano_naive[0:h2, 0:w2] = images[1] # 直接覆盖，产生明显接缝
                    
                    st.subheader("Blending 效果对比")
                    st.image(pano_naive, caption="Naive 拼接 (直接覆盖，无 Blending，可见明显接缝)", use_container_width=True)
                    
                    if status == cv2.Stitcher_OK:
                        st.image(pano_blended, caption="OpenCV Stitcher (内置 Multi-band Blending 融合，接缝平滑)", use_container_width=True)
                    else:
                        st.error(f"OpenCV 高级拼接失败，状态码: {status}")
                else:
                    st.error("特征点不足以进行对比拼接。")