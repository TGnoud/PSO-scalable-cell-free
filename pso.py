import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. CẤU HÌNH BÀI TOÁN (Hàm Sphere) ---
def objective_function(pos):
    """
    Hàm mục tiêu: f(x, y) = x^2 + y^2
    Đầu vào: pos là mảng [x, y]
    Đầu ra: giá trị z
    """
    x = pos[0]
    y = pos[1]
    return x**2 + y**2

# --- 2. THAM SỐ PSO  ---
w = 0.5    # Trọng số quán tính (Inertia Weight)
c1 = 1.5   # Hệ số nhận thức 
c2 = 1.5   # Hệ số xã hội 

n_particles = 30    # Số lượng hạt 
n_iterations = 50   # Số vòng lặp tối đa
bounds = [-10, 10]  # Giới hạn không gian tìm kiếm (-10 đến 10)

# --- 3. KHỞI TẠO ---
# Vị trí ngẫu nhiên của 30 hạt trong khoảng [-10, 10]
X = np.random.uniform(bounds[0], bounds[1], (n_particles, 2))
# Vận tốc ban đầu ngẫu nhiên
V = np.random.uniform(-1, 1, (n_particles, 2))

# Khởi tạo pbest và gbest
pbest_pos = X.copy()
pbest_val = np.array([objective_function(p) for p in X])

# Tìm gbest ban đầu
gbest_index = np.argmin(pbest_val)
gbest_pos = pbest_pos[gbest_index].copy()
gbest_val = pbest_val[gbest_index]

print(f"Khởi tạo: Gbest ban đầu tại {gbest_pos} với giá trị {gbest_val:.4f}")

# --- 4. VÒNG LẶP CHÍNH (Thuật toán PSO) ---
history = [] 

for i in range(n_iterations):
    # Lưu vị trí hiện tại để vẽ
    history.append(X.copy())
    
    # Tạo số ngẫu nhiên r1, r2 cho công thức
    r1 = np.random.rand(n_particles, 2)
    r2 = np.random.rand(n_particles, 2)
    
    # === CẬP NHẬT VẬN TỐC ===
    V = (w * V) + (c1 * r1 * (pbest_pos - X)) + (c2 * r2 * (gbest_pos - X))
    
    # === CẬP NHẬT VỊ TRÍ ===
    X = X + V
    
    # Giữ hạt trong biên giới hạn [-10, 10]
    X = np.clip(X, bounds[0], bounds[1])
    
    # === ĐÁNH GIÁ VÀ CẬP NHẬT Pbest, Gbest ===
    # Tính giá trị mục tiêu mới cho tất cả các hạt
    current_val = np.array([objective_function(p) for p in X])
    
    # Cập nhật Pbest 
    better_mask = current_val < pbest_val # So sánh xem vị trí mới có tốt hơn cũ không
    pbest_pos[better_mask] = X[better_mask]
    pbest_val[better_mask] = current_val[better_mask]
    
    # Cập nhật Gbest 
    min_val_idx = np.argmin(pbest_val)
    if pbest_val[min_val_idx] < gbest_val:
        gbest_val = pbest_val[min_val_idx]
        gbest_pos = pbest_pos[min_val_idx].copy()
        print(f"Vòng lặp {i+1}: Tìm thấy Gbest mới tốt hơn! -> {gbest_val:.30f} tại {gbest_pos}")

print("-" * 30)
print(f"KẾT QUẢ CUỐI CÙNG SAU {n_iterations} VÒNG LẶP:")
print(f"Vị trí tối ưu tìm được: {gbest_pos}")
print(f"Giá trị chính xác: {gbest_val:.30f}")

# --- 5. VẼ BIỂU ĐỒ ---
# Phần này sẽ hiển thị cửa sổ đồ họa các chấm xanh di chuyển vào tâm
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(bounds[0], bounds[1])
ax.set_ylim(bounds[0], bounds[1])
ax.set_title(f'Mô phỏng PSO - Vòng lặp 0')
scat = ax.scatter(history[0][:,0], history[0][:,1], c='blue', alpha=0.6)
ax.scatter(0, 0, c='red', marker='x', s=100, label='Đích (0,0)') # Điểm đích
ax.legend()

def update(frame):
    scat.set_offsets(history[frame])
    ax.set_title(f'PSO - Vòng lặp {frame}/{n_iterations} - Gbest: {objective_function(history[frame].mean(axis=0)):.2f}')
    return scat,

ani = FuncAnimation(fig, update, frames=len(history), interval=100, blit=True)
plt.show()