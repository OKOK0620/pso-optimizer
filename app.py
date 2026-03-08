import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import base64
from io import BytesIO

# 设置中文字体和负号显示
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------
# 1. 核心PSO算法（纯同步版，无线程/生成器）
# --------------------------
def pso_optimize(objective_func, dim=2, bounds=None, num_particles=200, max_iter=1000,
                 w_start=0.9, w_end=0.4, c1=2.05, c2=2.05,
                 convergence_threshold=1e-6, convergence_window=30, perturbation_rate=0.4):
    if bounds is None:
        bounds = [(-10, 10)] * dim

    # 初始化粒子
    particles = []
    for _ in range(num_particles):
        position = np.array([np.random.uniform(low, high) for (low, high) in bounds])
        velocity = np.array([np.random.uniform(-(high - low)/2, (high - low)/2) for (low, high) in bounds])
        current_val = objective_func(position)
        particles.append({
            'position': position,
            'velocity': velocity,
            'best_position': position.copy(),
            'best_value': current_val
        })

    # 初始化全局最优
    gbest_particle = min(particles, key=lambda p: p['best_value'])
    gbest_position = gbest_particle['best_position'].copy()
    gbest_value = gbest_particle['best_value']
    
    # 记录历史数据
    gbest_history = [gbest_value]          
    particle_history = []                 
    gbest_position_history = [gbest_position.copy()]

    # 收敛检测变量
    no_improvement_count = 0
    prev_best_value = gbest_value
    max_velocity = np.array([(high - low)*0.2 for (low, high) in bounds])

    # 迭代优化（纯同步，无进度反馈，避免云端兼容问题）
    for i in range(max_iter):
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

    # 最终返回所有结果
    return {
        'gbest_position': gbest_position,
        'gbest_value': gbest_value,
        'gbest_history': gbest_history,
        'particle_history': particle_history,
        'gbest_position_history': gbest_position_history
    }

# --------------------------
# 2. 可视化核心函数（简化版）
# --------------------------
def create_3d_animation(objective_func, bounds, particle_history, gbest_position_history, interval=50):
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # 创建网格数据
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x, y)
    
    # 计算Z值
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = objective_func([X[i,j], Y[i,j]])

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
        frame = min(frame, len(particle_history)-1)
        positions = particle_history[frame]
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        z_coords = np.array([objective_func([x, y]) for x, y in zip(x_coords, y_coords)])
        
        particles_scatter._offsets3d = (x_coords, y_coords, z_coords)
        
        gbest_pos = gbest_position_history[frame]
        gbest_z = objective_func(gbest_pos)
        gbest_scatter._offsets3d = ([gbest_pos[0]], [gbest_pos[1]], [gbest_z])
        
        return particles_scatter, gbest_scatter

    ani = FuncAnimation(fig, update, frames=len(particle_history), 
                        interval=interval, blit=False, repeat=False)
    return ani

def create_2d_animation(objective_func, bounds, particle_history, gbest_position_history, interval=50):
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # 创建网格数据
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # 计算Z值
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = objective_func([X[i,j], Y[i,j]])

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
        frame = min(frame, len(particle_history)-1)
        positions = particle_history[frame]
        particles_scatter.set_offsets(positions)
        
        gbest_pos = gbest_position_history[frame]
        gbest_point.set_data([gbest_pos[0]], [gbest_pos[1]])
        
        return particles_scatter, gbest_point

    ani = FuncAnimation(fig, update, frames=len(particle_history), 
                        interval=interval, blit=True, repeat=False)
    return ani

def save_animation_to_bytes(ani, fps=20):
    buf = BytesIO()
    ani.save(buf, writer='pillow', fps=fps, format='gif')
    buf.seek(0)
    return buf

def plot_convergence_curve(history):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history, color='darkblue', linewidth=2)
    ax.set_title('PSO算法收敛曲线', fontsize=14)
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('全局最优值', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    return fig

# --------------------------
# 3. Streamlit网页界面（纯同步版）
# --------------------------
def main():
    # 页面配置
    st.set_page_config(
        page_title="PSO 3D可视化优化器",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 页面样式
    st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; color: #2c3e50; font-weight: bold; margin-bottom: 1rem; }
    .sub-title { font-size: 1.2rem; color: #7f8c8d; margin-bottom: 2rem; }
    .result-card { background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stButton>button { background-color: #e74c3c; color: white; font-size: 1.1rem; border-radius: 8px; height: 3rem; }
    .stButton>button:hover { background-color: #c0392b; }
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
            value="100*sqrt(abs(y - 0.01*x**2)) + 0.01*abs(x + 10)",
            help="直接写数学表达式：sqrt/abs/sin/cos/tan/exp/log，示例：x**2 + y**2"
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
        num_particles = st.slider("粒子数量", 50, 500, 200)
        max_iter = st.slider("迭代次数", 100, 2000, 1000)
        c1 = st.slider("认知系数 c1", 0.1, 4.0, 2.05)
        c2 = st.slider("社会系数 c2", 0.1, 4.0, 2.05)
        perturbation_rate = st.slider("扰动粒子比例", 0.1, 0.8, 0.4)
        
        # 4. 可视化设置
        st.subheader("🎨 可视化设置")
        animation_speed = st.slider("动画速度 (ms/帧)", 10, 200, 50)
        show_3d = st.checkbox("显示3D动画", value=True)
        show_2d = st.checkbox("显示2D等高线动画", value=True)

    # 主页面计算按钮（纯同步，无线程）
    if st.button("🚀 启动PSO优化算法", use_container_width=True):
        try:
            # 1. 构建目标函数
            def objective_function(params):
                x, y = params[0], params[1]
                # 安全替换数学函数
                func_str = func_input.replace("sqrt", "np.sqrt")
                func_str = func_str.replace("abs", "np.abs")
                func_str = func_str.replace("sin", "np.sin")
                func_str = func_str.replace("cos", "np.cos")
                func_str = func_str.replace("tan", "np.tan")
                func_str = func_str.replace("exp", "np.exp")
                func_str = func_str.replace("log", "np.log")
                try:
                    return float(eval(func_str))
                except:
                    return float('inf')
            
            # 2. 显示加载状态
            with st.spinner("✅ 函数解析成功，正在优化中...（请稍等，约1-2分钟）"):
                # 3. 运行PSO算法（纯同步）
                pso_result = pso_optimize(
                    objective_function,
                    dim=2,
                    bounds=bounds,
                    num_particles=num_particles,
                    max_iter=max_iter,
                    c1=c1,
                    c2=c2,
                    perturbation_rate=perturbation_rate
                )
            
            # 4. 显示优化结果
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("🏆 优化结果")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最优X值", f"{pso_result['gbest_position'][0]:.6f}")
            with col2:
                st.metric("最优Y值", f"{pso_result['gbest_position'][1]:.6f}")
            with col3:
                st.metric("函数最小值", f"{pso_result['gbest_value']:.8f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 5. 显示收敛曲线
            st.subheader("📈 收敛曲线")
            conv_fig = plot_convergence_curve(pso_result['gbest_history'])
            st.pyplot(conv_fig)
            
            # 6. 生成并显示动画
            st.subheader("🎬 动态可视化结果")
            
            # 3D动画
            if show_3d and len(pso_result['particle_history']) > 0:
                with st.spinner("🌀 生成3D粒子运动动画..."):
                    ani_3d = create_3d_animation(objective_function, bounds, 
                                                pso_result['particle_history'], 
                                                pso_result['gbest_position_history'], 
                                                interval=animation_speed)
                    buf_3d = save_animation_to_bytes(ani_3d)
                    st.image(buf_3d, caption="PSO算法3D粒子运动过程", use_container_width=True)
                    
                    # 下载按钮
                    st.download_button(
                        label="📥 下载3D动画 (GIF)",
                        data=buf_3d,
                        file_name="pso_3d_animation.gif",
                        mime="image/gif",
                        use_container_width=True
                    )
            
            # 2D动画
            if show_2d and len(pso_result['particle_history']) > 0:
                with st.spinner("🌀 生成2D等高线动画..."):
                    ani_2d = create_2d_animation(objective_function, bounds, 
                                                pso_result['particle_history'], 
                                                pso_result['gbest_position_history'], 
                                                interval=animation_speed)
                    buf_2d = save_animation_to_bytes(ani_2d)
                    st.image(buf_2d, caption="PSO算法2D粒子运动过程", use_container_width=True)
                    
                    # 下载按钮
                    st.download_button(
                        label="📥 下载2D动画 (GIF)",
                        data=buf_2d,
                        file_name="pso_2d_animation.gif",
                        mime="image/gif",
                        use_container_width=True
                    )
            
            # 7. 结果导出
            st.subheader("💾 结果导出")
            result_text = f"""
            PSO优化结果
            ============
            目标函数：{func_input}
            定义域：X∈[{x_min}, {x_max}], Y∈[{y_min}, {y_max}]
            算法参数：粒子数={num_particles}, 迭代次数={max_iter}, c1={c1}, c2={c2}, 扰动比例={perturbation_rate}
            最优解：x={pso_result['gbest_position'][0]:.6f}, y={pso_result['gbest_position'][1]:.6f}
            函数最小值：{pso_result['gbest_value']:.8f}
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
            st.info("💡 正确示例：100*sqrt(abs(y - 0.01*x**2)) + 0.01*abs(x + 10)")

# 运行网页
if __name__ == "__main__":
    main()