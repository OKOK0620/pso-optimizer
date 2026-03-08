import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy.utilities.lambdify import lambdify
from matplotlib.animation import FuncAnimation
import base64
from io import BytesIO, StringIO
import os
import time

# 设置中文字体和负号显示
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'  # 确保ffmpeg可用（可选）

# --------------------------
# 1. 核心PSO算法（保留改进版逻辑）
# --------------------------
def pso_optimize(objective_func, dim=2, bounds=None, num_particles=200, max_iter=1000,
                 w_start=0.9, w_end=0.4, c1=2.05, c2=2.05, verbose=False,
                 convergence_threshold=1e-6, convergence_window=30, perturbation_rate=0.4):
    """改进版PSO算法，记录粒子轨迹用于3D可视化"""
    if bounds is None:
        bounds = [(-10, 10)] * dim

    # 初始化粒子
    particles = []
    for _ in range(num_particles):
        position = np.array([np.random.uniform(low, high) for (low, high) in bounds])
        velocity = np.array([np.random.uniform(-(high - low)/2, (high - low)/2) for (low, high) in bounds])
        particles.append({
            'position': position,
            'velocity': velocity,
            'best_position': position.copy(),
            'best_value': objective_func(position)
        })

    # 初始化全局最优
    gbest_particle = min(particles, key=lambda p: p['best_value'])
    gbest_position = gbest_particle['best_position'].copy()
    gbest_value = gbest_particle['best_value']
    
    # 记录历史数据（用于可视化）
    gbest_history = [gbest_value]          # 全局最优值历史
    particle_history = []                 # 每代粒子位置
    gbest_position_history = [gbest_position.copy()]  # 全局最优位置历史

    # 收敛检测变量
    no_improvement_count = 0
    prev_best_value = gbest_value
    max_velocity = np.array([(high - low)*0.2 for (low, high) in bounds])

    # 迭代优化
    for i in range(max_iter):
        # 动态惯性权重
        w = w_start - (w_start - w_end) * (i / max_iter)
        current_positions = []

        # 更新每个粒子
        for particle in particles:
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            # 速度更新
            cognitive = c1 * r1 * (particle['best_position'] - particle['position'])
            social = c2 * r2 * (gbest_position - particle['position'])
            new_velocity = w * particle['velocity'] + cognitive + social
            
            # 速度限制
            new_velocity = np.clip(new_velocity, -max_velocity, max_velocity)
            
            # 位置更新
            particle['velocity'] = new_velocity
            particle['position'] += new_velocity
            
            # 边界检查
            particle['position'] = np.clip(particle['position'], 
                                         [b[0] for b in bounds], 
                                         [b[1] for b in bounds])
            
            # 更新个体最优
            current_value = objective_func(particle['position'])
            if current_value < particle['best_value']:
                particle['best_value'] = current_value
                particle['best_position'] = particle['position'].copy()
            
            current_positions.append(particle['position'].copy())

        # 记录当前代粒子位置
        particle_history.append(np.array(current_positions))
        
        # 更新全局最优
        current_best_particle = min(particles, key=lambda p: p['best_value'])
        if current_best_particle['best_value'] < gbest_value:
            gbest_position = current_best_particle['best_position'].copy()
            gbest_value = current_best_particle['best_value']
            no_improvement_count = 0
        else:
            if abs(gbest_value - prev_best_value) < convergence_threshold:
                no_improvement_count += 1
            prev_best_value = gbest_value

        # 早熟收敛处理
        if no_improvement_count >= convergence_window:
            num_perturb = int(num_particles * perturbation_rate)
            perturb_indices = np.random.choice(num_particles, num_perturb, replace=False)
            for idx in perturb_indices:
                particles[idx]['position'] = np.array([np.random.uniform(low, high) for (low, high) in bounds])
                particles[idx]['velocity'] = np.array([np.random.uniform(-(high - low)/2, (high - low)/2) for (low, high) in bounds])
                particles[idx]['best_position'] = particles[idx]['position'].copy()
                particles[idx]['best_value'] = objective_func(particles[idx]['position'])
            no_improvement_count = 0

        # 记录全局最优历史
        gbest_history.append(gbest_value)
        gbest_position_history.append(gbest_position.copy())

        # 进度反馈（每100代）
        if verbose and i % 100 == 0:
            yield i, max_iter, gbest_value, gbest_position  # 生成器返回进度

    # 最终返回结果
    yield max_iter, max_iter, gbest_value, gbest_position  # 完成标记
    return gbest_position, gbest_value, gbest_history, particle_history, gbest_position_history

# --------------------------
# 2. 可视化核心函数（新增3D动态）
# --------------------------
def create_3d_animation(objective_func, bounds, particle_history, gbest_position_history, interval=50):
    """生成3D动态粒子运动动画"""
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # 创建网格数据
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.array([objective_func([xi, yi]) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    # 创建3D画布
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制函数表面
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # 初始化粒子和全局最优标记
    particles_scatter = ax.scatter([], [], [], c='red', s=15, alpha=0.7, label='粒子群')
    gbest_scatter = ax.scatter([], [], [], c='yellow', s=100, marker='*', label='全局最优')
    
    # 设置坐标轴
    ax.set_xlabel('X轴', fontsize=12)
    ax.set_ylabel('Y轴', fontsize=12)
    ax.set_zlabel('函数值 f(x,y)', fontsize=12)
    ax.set_title('PSO算法3D粒子运动过程', fontsize=14)
    ax.legend()

    # 更新函数
    def update(frame):
        if frame >= len(particle_history):
            frame = len(particle_history) - 1
        
        # 获取当前帧粒子位置
        positions = particle_history[frame]
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        z_coords = np.array([objective_func([x, y]) for x, y in zip(x_coords, y_coords)])
        
        # 更新粒子位置
        particles_scatter._offsets3d = (x_coords, y_coords, z_coords)
        
        # 更新全局最优位置
        gbest_pos = gbest_position_history[frame]
        gbest_z = objective_func(gbest_pos)
        gbest_scatter._offsets3d = ([gbest_pos[0]], [gbest_pos[1]], [gbest_z])
        
        return particles_scatter, gbest_scatter

    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(particle_history), 
                        interval=interval, blit=False, repeat=False)
    return ani

def create_2d_animation(objective_func, bounds, particle_history, gbest_position_history, interval=50):
    """生成2D等高线动态图（保留原有功能）"""
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # 创建网格数据
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([objective_func([xi, yi]) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    # 创建2D画布
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.7)
    ax.contour(X, Y, Z, 10, colors='black', linewidths=0.5)
    plt.colorbar(contour, ax=ax)
    
    # 初始化粒子和全局最优标记
    particles_scatter = ax.scatter([], [], c='red', s=20, alpha=0.6, label='粒子群')
    gbest_point, = ax.plot([], [], 'y*', markersize=15, label='全局最优')
    
    # 设置坐标轴
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X轴', fontsize=12)
    ax.set_ylabel('Y轴', fontsize=12)
    ax.set_title('PSO算法2D粒子运动过程', fontsize=14)
    ax.legend()

    # 更新函数
    def update(frame):
        if frame >= len(particle_history):
            frame = len(particle_history) - 1
        
        # 更新粒子位置
        positions = particle_history[frame]
        particles_scatter.set_offsets(positions)
        
        # 更新全局最优位置
        gbest_pos = gbest_position_history[frame]
        gbest_point.set_data([gbest_pos[0]], [gbest_pos[1]])
        
        return particles_scatter, gbest_point

    ani = FuncAnimation(fig, update, frames=len(particle_history), 
                        interval=interval, blit=True, repeat=False)
    return ani

def save_animation_to_bytes(ani, format='gif', fps=20):
    """将动画保存为字节流（用于网页展示）"""
    buf = BytesIO()
    ani.save(buf, writer='pillow', fps=fps, format=format)
    buf.seek(0)
    return buf

def plot_convergence_curve(history):
    """绘制收敛曲线"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history, color='darkblue', linewidth=2)
    ax.set_title('PSO算法收敛曲线', fontsize=14)
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('全局最优值', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # 对数刻度更易观察收敛
    return fig

# --------------------------
# 3. Streamlit网页界面（升级版）
# --------------------------
def main():
    # 页面配置
    st.set_page_config(
        page_title="PSO 3D可视化优化器",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 页面标题和样式
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">粒子群算法（PSO）3D可视化优化器</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">高精度全局极小值搜索 | 3D动态粒子运动可视化 | 支持自定义函数</div>', unsafe_allow_html=True)

    # 侧边栏参数面板
    with st.sidebar:
        st.header("⚙️ 算法参数配置")
        
        # 1. 函数输入
        st.subheader("🎯 目标函数")
        func_input = st.text_input(
            "输入二元函数 f(x,y)",
            value="100*sqrt(abs(y-0.01*x**2)) + 0.01*abs(x + 10)",
            help="支持numpy语法：sqrt/abs/pow/sin/cos等，示例：x**2 + y**2 或 sin(x) + cos(y)"
        )
        
        # 2. 定义域设置
        st.subheader("📏 定义域范围")
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.number_input("X最小值", value=-15.0, step=0.5)
            y_min = st.number_input("Y最小值", value=-5.0, step=0.5)
        with col2:
            x_max = st.number_input("X最大值", value=5.0, step=0.5)
            y_max = st.number_input("Y最大值", value=15.0, step=0.5)
        bounds = [(x_min, x_max), (y_min, y_max)]
        
        # 3. 算法参数
        st.subheader("🧬 PSO算法参数")
        num_particles = st.slider("粒子数量", 50, 500, 200, help="数量越多越精准，但速度越慢")
        max_iter = st.slider("迭代次数", 100, 2000, 1000, help="迭代越多收敛越充分")
        c1 = st.slider("认知系数 c1", 0.1, 4.0, 2.05, help="粒子向自身最优学习的权重")
        c2 = st.slider("社会系数 c2", 0.1, 4.0, 2.05, help="粒子向全局最优学习的权重")
        perturbation_rate = st.slider("扰动粒子比例", 0.1, 0.8, 0.4, help="收敛时重新初始化的粒子比例")
        
        # 4. 可视化设置
        st.subheader("🎨 可视化设置")
        animation_speed = st.slider("动画速度 (ms/帧)", 10, 200, 50)
        show_3d = st.checkbox("显示3D动画", value=True)
        show_2d = st.checkbox("显示2D等高线动画", value=True)

    # 主页面计算按钮
    if st.button("🚀 启动PSO优化算法", type="primary", use_container_width=True):
        try:
            # 1. 解析目标函数
            x, y = sp.symbols('x y')
            func_sym = eval(func_input)
            func = lambdify((x, y), func_sym, 'numpy')
            
            def objective_function(params):
                """适配PSO的目标函数格式"""
                return func(params[0], params[1])
            
            st.success("✅ 函数解析成功，开始优化...")
            
            # 2. 显示进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            result_placeholder = st.empty()
            
            # 3. 运行PSO算法（带进度反馈）
            pso_generator = pso_optimize(
                objective_function,
                dim=2,
                bounds=bounds,
                num_particles=num_particles,
                max_iter=max_iter,
                c1=c1,
                c2=c2,
                perturbation_rate=perturbation_rate,
                verbose=True
            )
            
            # 迭代获取进度
            final_gbest_pos = None
            final_gbest_val = None
            final_history = None
            final_particle_history = None
            final_gbest_pos_history = None
            
            for current_iter, total_iter, current_val, current_pos in pso_generator:
                # 更新进度
                progress = current_iter / total_iter
                progress_bar.progress(progress)
                status_text.text(f"🔄 迭代中：{current_iter}/{total_iter} | 当前最优值：{current_val:.8f} | 最优位置：[{current_pos[0]:.4f}, {current_pos[1]:.4f}]")
                
                # 最后一次返回完整结果
                if current_iter == total_iter:
                    final_gbest_pos = current_pos
                    final_gbest_val = current_val
                    
                    # 重新运行获取完整历史（生成器只能遍历一次）
                    _, final_gbest_val, final_history, final_particle_history, final_gbest_pos_history = list(pso_generator)[-1]
            
            # 4. 隐藏进度条，显示结果
            progress_bar.empty()
            status_text.empty()
            
            with result_placeholder.container():
                # 显示优化结果
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("🏆 优化结果")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最优X值", f"{final_gbest_pos[0]:.6f}")
                with col2:
                    st.metric("最优Y值", f"{final_gbest_pos[1]:.6f}")
                with col3:
                    st.metric("函数最小值", f"{final_gbest_val:.8f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 显示收敛曲线
                st.subheader("📈 收敛曲线")
                conv_fig = plot_convergence_curve(final_history)
                st.pyplot(conv_fig)
                
                # 生成并显示动画
                st.subheader("🎬 动态可视化结果")
                
                # 3D动画
                if show_3d:
                    with st.spinner("🌀 生成3D粒子运动动画..."):
                        ani_3d = create_3d_animation(objective_function, bounds, final_particle_history, final_gbest_pos_history, interval=animation_speed)
                        buf_3d = save_animation_to_bytes(ani_3d)
                        st.image(buf_3d, caption="PSO算法3D粒子运动过程", use_container_width=True)
                        
                        # 提供下载
                        b64_3d = base64.b64encode(buf_3d.getvalue()).decode()
                        st.download_button(
                            label="📥 下载3D动画 (GIF)",
                            data=buf_3d,
                            file_name="pso_3d_animation.gif",
                            mime="image/gif",
                            use_container_width=True
                        )
                
                # 2D动画
                if show_2d:
                    with st.spinner("🌀 生成2D等高线动画..."):
                        ani_2d = create_2d_animation(objective_function, bounds, final_particle_history, final_gbest_pos_history, interval=animation_speed)
                        buf_2d = save_animation_to_bytes(ani_2d)
                        st.image(buf_2d, caption="PSO算法2D粒子运动过程", use_container_width=True)
                        
                        # 提供下载
                        b64_2d = base64.b64encode(buf_2d.getvalue()).decode()
                        st.download_button(
                            label="📥 下载2D动画 (GIF)",
                            data=buf_2d,
                            file_name="pso_2d_animation.gif",
                            mime="image/gif",
                            use_container_width=True
                        )
                
                # 结果导出
                st.subheader("💾 结果导出")
                result_text = f"""
                PSO优化结果
                ============
                目标函数：{func_input}
                定义域：X∈[{x_min}, {x_max}], Y∈[{y_min}, {y_max}]
                算法参数：粒子数={num_particles}, 迭代次数={max_iter}, c1={c1}, c2={c2}, 扰动比例={perturbation_rate}
                最优解：x={final_gbest_pos[0]:.6f}, y={final_gbest_pos[1]:.6f}
                函数最小值：{final_gbest_val:.8f}
                """
                st.download_button(
                    label="📄 导出优化结果 (TXT)",
                    data=result_text,
                    file_name="pso_optimization_result.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ 运行出错：{str(e)}")
            st.info("💡 提示：请检查函数表达式是否正确（如使用abs()而非math.abs，sqrt()而非math.sqrt）")

# 运行网页
if __name__ == "__main__":
    main()