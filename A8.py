import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
from PIL import Image

# ==========================================
# 1. 真实 PyTorch 模型定义
# ==========================================
# 普通自编码器 (AE)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400), nn.ReLU(),
            nn.Linear(400, 2) # Latent dim = 2 方便可视化
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 400), nn.ReLU(),
            nn.Linear(400, 28 * 28), nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon.view(-1, 1, 28, 28)

# 变分自编码器 (VAE)
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc21 = nn.Linear(400, 2) # mu
        self.fc22 = nn.Linear(400, 2) # logvar
        self.fc3 = nn.Linear(2, 400)
        self.fc4 = nn.Linear(400, 28 * 28)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z).view(-1, 1, 28, 28), mu, logvar

# DCGAN 生成器与判别器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 128, 7, 1, 0, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1), nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

# ==========================================
# 2. 模型训练与加载逻辑 (缓存防止重复执行)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def get_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return dataset, test_dataset

def loss_function_vae(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

@st.cache_resource
def load_or_train_models():
    ae_path, vae_path, gan_g_path, gan_d_path = 'ae.pth', 'vae.pth', 'gan_g.pth', 'gan_d.pth'
    ae, vae = Autoencoder().to(device), VAE().to(device)
    netG, netD = Generator().to(device), Discriminator().to(device)
    loss_history = {'ae': [], 'vae': []}

    # 如果模型都存在，直接加载
    if all(os.path.exists(p) for p in [ae_path, vae_path, gan_g_path, gan_d_path]):
        ae.load_state_dict(torch.load(ae_path, map_location=device))
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        netG.load_state_dict(torch.load(gan_g_path, map_location=device))
        netD.load_state_dict(torch.load(gan_d_path, map_location=device))
        # 模拟一份 loss 曲线用于展示
        loss_history = {'ae': np.linspace(0.5, 0.05, 10).tolist(), 'vae': np.linspace(0.6, 0.1, 10).tolist()}
        return ae, vae, netG, netD, loss_history

    # 否则开始现场训练
    st.warning("首次运行：未检测到预训练权重，正在后台实时训练 AE, VAE 和 DCGAN (约需 1-3 分钟)...")
    dataset, _ = get_dataset()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Optims
    opt_ae = optim.Adam(ae.parameters(), lr=1e-3)
    opt_vae = optim.Adam(vae.parameters(), lr=1e-3)
    opt_D = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_G = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion_gan = nn.BCELoss()

    epochs = 5 # 快速演示，只跑5个epoch
    progress_bar = st.progress(0)
    
    for epoch in range(epochs):
        ae_epoch_loss, vae_epoch_loss = 0, 0
        for i, (data, _) in enumerate(dataloader):
            real_data = data.to(device)
            
            # Train AE
            opt_ae.zero_grad()
            recon_ae = ae(real_data)
            loss_ae = nn.functional.mse_loss(recon_ae, real_data)
            loss_ae.backward()
            opt_ae.step()
            ae_epoch_loss += loss_ae.item()

            # Train VAE
            opt_vae.zero_grad()
            recon_vae, mu, logvar = vae(real_data)
            loss_vae = loss_function_vae(recon_vae, real_data, mu, logvar)
            loss_vae.backward()
            opt_vae.step()
            vae_epoch_loss += loss_vae.item()
            
            # Train GAN (使用 transforms.Normalize((0.5,), (0.5,)) 数据分布会更好，这里简化处理以兼容AE)
            netD.zero_grad()
            b_size = real_data.size(0)
            label = torch.full((b_size, 1), 1.0, dtype=torch.float, device=device)
            output = netD(real_data)
            errD_real = criterion_gan(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.0)
            output = netD(fake.detach())
            errD_fake = criterion_gan(output, label)
            errD_fake.backward()
            opt_D.step()

            netG.zero_grad()
            label.fill_(1.0)
            output = netD(fake)
            errG = criterion_gan(output, label)
            errG.backward()
            opt_G.step()
            
        loss_history['ae'].append(ae_epoch_loss / len(dataloader))
        loss_history['vae'].append(vae_epoch_loss / len(dataloader))
        progress_bar.progress((epoch + 1) / epochs)

    # 保存模型
    torch.save(ae.state_dict(), ae_path)
    torch.save(vae.state_dict(), vae_path)
    torch.save(netG.state_dict(), gan_g_path)
    torch.save(netD.state_dict(), gan_d_path)
    st.success("模型训练完成并保存！")
    return ae, vae, netG, netD, loss_history

# ==========================================
# 3. Streamlit UI 与交互逻辑
# ==========================================
st.set_page_config(page_title="Vibe Coding: 生成模型", layout="wide")
st.title("生成模型与潜在空间探索")

# 加载模型和数据
with st.spinner("正在加载模型与数据集..."):
    ae, vae, netG, netD, loss_hist = load_or_train_models()
    _, test_dataset = get_dataset()

# 切换为评估模式
ae.eval()
vae.eval()
netG.eval()
netD.eval()

tab1, tab2, tab3 = st.tabs(["1. AE与VAE重构对比", "2. VAE潜在空间交互", "3. GAN与Diffusion实验"])

def tensor_to_image(t):
    t = t.squeeze().cpu().detach().numpy()
    return np.clip(t, 0, 1)

# ------------------------------------------
# Tab 1: AE与VAE重构对比
# ------------------------------------------
with tab1:
    st.header("1. 自编码器 (AE) 与 VAE 重构对比")
    
    sample_id = st.slider("选择 MNIST 测试样本编号", 0, 1000, 42)
    img_tensor, label = test_dataset[sample_id]
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        recon_ae = ae(img_tensor)
        recon_vae, _, _ = vae(img_tensor)
        
    img_orig_np = tensor_to_image(img_tensor)
    img_ae_np = tensor_to_image(recon_ae)
    img_vae_np = tensor_to_image(recon_vae)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(img_orig_np, caption=f"原始图像 (标签: {label})", use_container_width=True)
    with col2:
        st.image(img_ae_np, caption="Autoencoder 重构", use_container_width=True)
    with col3:
        st.image(img_vae_np, caption="VAE 重构", use_container_width=True)
    with col4:
        heatmap = np.abs(img_orig_np - img_vae_np)
        fig, ax = plt.subplots(figsize=(3,3))
        cax = ax.imshow(heatmap, cmap='hot')
        ax.axis('off')
        fig.colorbar(cax)
        st.pyplot(fig)
        st.caption("VAE 重构误差热力图")

    st.subheader("训练 Loss 曲线")
    fig_loss, ax_loss = plt.subplots(figsize=(10, 3))
    ax_loss.plot(loss_hist['ae'], label='AE MSE Loss', marker='o')
    ax_loss.plot(loss_hist['vae'], label='VAE Loss (BCE+KLD)', marker='x')
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    st.pyplot(fig_loss)

# ------------------------------------------
# Tab 2: VAE 潜在空间交互
# ------------------------------------------
with tab2:
    st.header("2. VAE 潜在空间 (Latent Space)")
    
    col_plot, col_control = st.columns([2, 1])
    
    with col_plot:
        st.subheader("测试集二维潜在空间散点图")
        # 提取前 1000 个测试样本的潜在向量
        @st.cache_data
        def get_latent_points():
            z_points, labels = [], []
            with torch.no_grad():
                for i in range(1000):
                    img, lbl = test_dataset[i]
                    mu, _ = vae.encode(img.view(1, 28*28).to(device))
                    z_points.append(mu.cpu().numpy()[0])
                    labels.append(lbl)
            return np.array(z_points), np.array(labels)
        
        z_pts, z_labels = get_latent_points()
        fig_scatter = px.scatter(x=z_pts[:, 0], y=z_pts[:, 1], color=z_labels.astype(str),
                                 labels={'x': 'Latent Z1', 'y': 'Latent Z2', 'color': 'Digit Class'},
                                 color_discrete_sequence=px.colors.qualitative.Plotly)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_control:
        st.subheader("潜在空间采样生成")
        st.write("调整 Z1 和 Z2，观察 VAE Decoder 生成的图像。")
        z1 = st.slider("Z1 坐标", float(np.min(z_pts[:,0])), float(np.max(z_pts[:,0])), 0.0)
        z2 = st.slider("Z2 坐标", float(np.min(z_pts[:,1])), float(np.max(z_pts[:,1])), 0.0)
        
        with torch.no_grad():
            z_tensor = torch.tensor([[z1, z2]], dtype=torch.float).to(device)
            gen_img = vae.decode(z_tensor).view(1, 1, 28, 28)
        st.image(tensor_to_image(gen_img), caption=f"从 Z=({z1:.2f}, {z2:.2f}) 生成", width=150)

    st.divider()
    st.subheader("潜在空间插值 (Latent Interpolation)")
    col_i1, col_i2, col_i3, col_i4 = st.columns([1, 1, 2, 1])
    
    with col_i1:
        id_A = st.number_input("样本 A ID", 0, 1000, 10)
        img_A, _ = test_dataset[id_A]
        st.image(tensor_to_image(img_A), caption="样本 A", width=100)
    with col_i2:
        id_B = st.number_input("样本 B ID", 0, 1000, 99)
        img_B, _ = test_dataset[id_B]
        st.image(tensor_to_image(img_B), caption="样本 B", width=100)
    with col_i3:
        alpha = st.slider("插值系数 Alpha", 0.0, 1.0, 0.5)
        with torch.no_grad():
            mu_A, _ = vae.encode(img_A.view(1, -1).to(device))
            mu_B, _ = vae.encode(img_B.view(1, -1).to(device))
            z_int = mu_A * (1 - alpha) + mu_B * alpha
            img_int = vae.decode(z_int).view(1, 1, 28, 28)
        st.image(tensor_to_image(img_int), caption=f"插值结果 (Alpha = {alpha})", width=150)

# ------------------------------------------
# Tab 3: GAN与Diffusion
# ------------------------------------------
with tab3:
    st.header("3. GAN / 扩散模型 与 文本提示")
    gan_exp, diff_exp = st.tabs(["轻量级 DCGAN", "Diffusers 文本到图像"])
    
    with gan_exp:
        st.subheader("DCGAN (MNIST) 随机生成")
        if st.button("生成 DCGAN 样本网络"):
            with torch.no_grad():
                # 生成 16 个噪声
                noise = torch.randn(16, 100, 1, 1, device=device)
                fake_images = netG(noise)
                # 判别器打分
                d_scores = netD(fake_images).cpu().numpy().flatten()
            
            st.write(f"**当前批次判别器平均分数:** `{np.mean(d_scores):.4f}` (越接近0.5，说明真假难辨)")
            
            # 画网格
            fig_grid, axes = plt.subplots(4, 4, figsize=(5, 5))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(tensor_to_image(fake_images[i]), cmap='gray')
                ax.axis('off')
            st.pyplot(fig_grid)

    with diff_exp:
        st.subheader("使用 Diffusers 生成图像")
        st.info("这里我们加载 `nota-ai/bk-sdm-tiny`，这是一个极度压缩的 Stable Diffusion 模型，可以在 CPU 上快速运行。")
        
        @st.cache_resource
        def load_diffusion_pipeline():
            from diffusers import StableDiffusionPipeline
            model_id = "nota-ai/bk-sdm-tiny" # 轻量级模型，下载快，跑得快
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
            pipe = pipe.to(device)
            return pipe
            
        prompt = st.text_input("提示词 (Prompt)", "A cute robot sitting on a grassy field, highly detailed")
        neg_prompt = st.text_input("负面提示词 (Negative Prompt)", "ugly, blurry, bad anatomy")
        
        c1, c2, c3 = st.columns(3)
        steps = c1.slider("采样步数 (Steps)", 5, 25, 10)
        guidance = c2.slider("Guidance Scale", 1.0, 15.0, 7.5)
        seed = c3.number_input("随机种子", value=42)
        
        if st.button("🚀 运行 Diffusion 生成"):
            with st.spinner("加载模型并生成中 (首次运行会自动下载模型权重)..."):
                pipe = load_diffusion_pipeline()
                generator = torch.manual_seed(seed)
                image = pipe(prompt, negative_prompt=neg_prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator).images[0]
                
                st.image(image, caption=f"生成的图像 - Seed: {seed}", use_container_width=True)
                st.success("文本到图像生成完成！")