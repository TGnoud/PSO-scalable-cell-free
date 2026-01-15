import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, toeplitz
from scipy.integrate import quad

# =============================================================================
# PHẦN 1: CÁC HÀM MÔ PHỎNG HỆ THỐNG (GIỮ NGUYÊN TỪ PROJECT CŨ)
# Nguồn: ga_scf.py
# =============================================================================

def db2pow(db):
    return 10**(db/10)

def functionRlocalscattering(M, theta, ASDdeg, antennaSpacing=0.5):
    ASD = ASDdeg * np.pi / 180
    firstRow = np.zeros(M, dtype=complex)
    for col in range(M):
        distance = antennaSpacing * col
        def integrand_real(Delta):
            return np.cos(2*np.pi*distance*np.sin(theta+Delta)) * np.exp(-Delta**2/(2*ASD**2))/(np.sqrt(2*np.pi)*ASD)
        def integrand_imag(Delta):
            return np.sin(2*np.pi*distance*np.sin(theta+Delta)) * np.exp(-Delta**2/(2*ASD**2))/(np.sqrt(2*np.pi)*ASD)
        val_real, _ = quad(integrand_real, -20*ASD, 20*ASD)
        val_imag, _ = quad(integrand_imag, -20*ASD, 20*ASD)
        firstRow[col] = val_real + 1j*val_imag
    R = toeplitz(firstRow)
    return R

def generateSetup(L, K, N, tau_p):
    # ... (Code giữ nguyên như ga_scf.py để đảm bảo tính nhất quán)
    squareLength = 2000
    B_bw = 20e6
    noiseFigure = 7
    noiseVariancedBm = -174 + 10*np.log10(B_bw) + noiseFigure
    alpha = 3.76
    sigma_sf = 10
    constantTerm = -35.3
    antennaSpacing = 0.5
    ASDdeg = 20
    threshold = -40

    APpositions = (np.random.rand(L) + 1j*np.random.rand(L)) * squareLength
    UEpositions = (np.random.rand(K) + 1j*np.random.rand(K)) * squareLength

    wrapHorizontal = np.array([-squareLength, 0, squareLength])
    wrapVertical = wrapHorizontal
    wrapLocations = (wrapHorizontal[:, None] + 1j*wrapVertical).flatten()
    APpositionsWrapped = APpositions[:, None] + wrapLocations
    
    dist_matrix = np.zeros((L, K))
    gainOverNoisedB = np.zeros((L, K))
    R = np.zeros((N, N, L, K), dtype=complex)
    D = np.zeros((L, K))
    pilotIndex = np.zeros(K, dtype=int)
    masterAPs = np.zeros(K, dtype=int)

    for k in range(K):
        dists = np.abs(APpositionsWrapped - UEpositions[k])
        min_dist_indices = np.argmin(dists, axis=1)
        distances = np.sqrt(10**2 + np.take_along_axis(dists, min_dist_indices[:, None], axis=1).flatten()**2)
        dist_matrix[:, k] = distances
        gainOverNoisedB[:, k] = constantTerm - alpha*10*np.log10(distances) + sigma_sf*np.random.randn(L) - noiseVariancedBm
        master = np.argmax(gainOverNoisedB[:, k])
        D[master, k] = 1
        masterAPs[k] = master
        
        if k < tau_p:
            pilotIndex[k] = k
        else:
            pilotInterference = np.zeros(tau_p)
            for t in range(tau_p):
                gains_of_existing_users = gainOverNoisedB[master, :k]
                mask_pilot_t = (pilotIndex[:k] == t)
                pilotInterference[t] = np.sum(db2pow(gains_of_existing_users[mask_pilot_t]))
            pilotIndex[k] = np.argmin(pilotInterference)
            
        for l in range(L):
            best_pos = min_dist_indices[l]
            angletoUE = np.angle(UEpositions[k] - APpositionsWrapped[l, best_pos])
            R[:, :, l, k] = db2pow(gainOverNoisedB[l, k]) * functionRlocalscattering(N, angletoUE, ASDdeg, antennaSpacing)

    for l in range(L):
        for t in range(tau_p):
            pilotUEs = np.where(pilotIndex == t)[0]
            if np.sum(D[l, pilotUEs]) == 0 and len(pilotUEs) > 0:
                gains = gainOverNoisedB[l, pilotUEs]
                best_idx = np.argmax(gains)
                bestUE = pilotUEs[best_idx]
                if gains[best_idx] - gainOverNoisedB[masterAPs[bestUE], bestUE] >= threshold:
                    D[l, bestUE] = 1
    return gainOverNoisedB, R, pilotIndex, D

def functionChannelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p):
    # ... (Giữ nguyên từ ga_scf.py)
    H = (np.random.randn(L*N, nbrOfRealizations, K) + 1j*np.random.randn(L*N, nbrOfRealizations, K)) * np.sqrt(0.5)
    for l in range(L):
        for k in range(K):
            Rsqrt = sqrtm(R[:, :, l, k])
            H[l*N:(l+1)*N, :, k] = Rsqrt @ H[l*N:(l+1)*N, :, k]
            
    Hhat = np.zeros_like(H)
    B = np.zeros_like(R)
    C = np.zeros_like(R)
    Np = (np.random.randn(N, nbrOfRealizations, L, tau_p) + 1j*np.random.randn(N, nbrOfRealizations, L, tau_p)) * np.sqrt(0.5)
    eyeN = np.eye(N)
    
    for l in range(L):
        for t in range(tau_p):
            pilotUEs = np.where(pilotIndex == t)[0]
            yp = np.sqrt(p)*tau_p * np.sum(H[l*N:(l+1)*N, :, pilotUEs], axis=2) + np.sqrt(tau_p)*Np[:, :, l, t]
            Psi = p*tau_p * np.sum(R[:, :, l, pilotUEs], axis=2) + eyeN
            PsiInv = np.linalg.inv(Psi)
            for k in pilotUEs:
                RPsi = R[:, :, l, k] @ PsiInv
                Hhat[l*N:(l+1)*N, :, k] = np.sqrt(p) * RPsi @ yp
                B[:, :, l, k] = p*tau_p * RPsi @ R[:, :, l, k]
                C[:, :, l, k] = R[:, :, l, k] - B[:, :, l, k]
    return Hhat, H, B, C

def compute_se_downlink_fast(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho_dist, R, pilotIndex, rho_central):
    # ... (Giữ nguyên logic tính SE P-MMSE từ ga_scf.py)
    prelogFactor = (1 - tau_p/tau_c)
    
    # Tính MR (để vẽ biểu đồ so sánh)
    signal_MR = np.zeros(K)
    interf_MR = np.zeros(K)
    cont_MR = np.zeros((K, K))
    for l in range(L):
        servedUEs = np.where(D[l, :] == 1)[0]
        for k in servedUEs:
            signal_MR[k] += np.sqrt(rho_dist[l, k] * np.trace(B[:, :, l, k].real))
            for i in range(K):
                term = rho_dist[l, k] * np.trace(B[:, :, l, k] @ R[:, :, l, i]).real / np.trace(B[:, :, l, k].real)
                interf_MR[i] += term
                if pilotIndex[k] == pilotIndex[i]:
                    cont_MR[i, k] += np.sqrt(rho_dist[l, k]) * np.trace((B[:, :, l, k] @ np.linalg.inv(R[:, :, l, k])) @ R[:, :, l, i]).real / np.sqrt(np.trace(B[:, :, l, k].real))
    SE_MR = prelogFactor * np.log2(1 + (np.abs(signal_MR)**2) / (interf_MR + np.sum(np.abs(cont_MR)**2, axis=1) - np.abs(signal_MR)**2 + 1))
    
    # Tính P-MMSE (Dùng rho_central được tối ưu bởi PSO)
    signal_P_MMSE = np.zeros(K)
    interf_P_MMSE_vec = np.zeros(K)
    
    for n in range(nbrOfRealizations):
        interf_P_MMSE_n_realization = np.zeros((K, K), dtype=complex)
        for k in range(K):
            servingAPs = np.where(D[:, k] == 1)[0]
            La = len(servingAPs)
            if La == 0: continue

            servedUEs_in_cluster = np.where(np.sum(D[servingAPs, :], axis=0) >= 1)[0]
            
            Hhat_active = np.zeros((N*La, K), dtype=complex)
            H_active = np.zeros((N*La, K), dtype=complex)
            C_tot_partial = np.zeros((N*La, N*La), dtype=complex)
            
            idx_start = 0
            for l_idx in range(La):
                l = servingAPs[l_idx]
                Hhat_active[idx_start:idx_start+N, :] = Hhat[l*N:(l+1)*N, n, :]
                H_active[idx_start:idx_start+N, :] = H[l*N:(l+1)*N, n, :]
                C_tot_partial[idx_start:idx_start+N, idx_start:idx_start+N] = np.sum(C[:, :, l, servedUEs_in_cluster], axis=2)
                idx_start += N
            
            try:
                inv_term = np.linalg.inv(p * (Hhat_active[:, servedUEs_in_cluster] @ Hhat_active[:, servedUEs_in_cluster].conj().T + C_tot_partial) + np.eye(La*N))
                w = p * inv_term @ Hhat_active[:, k]
            except np.linalg.LinAlgError:
                w = np.zeros(N*La, dtype=complex)

            norm_w = np.linalg.norm(w)
            if norm_w > 0:
                # Áp dụng công suất từ PSO
                w = w / norm_w * np.sqrt(rho_central[k]) 
            
            h_k = H_active[:, k]
            signal_P_MMSE[k] += (h_k.conj().T @ w).real / nbrOfRealizations
            
            for i in range(K):
                h_i = H_active[:, i]
                interf_P_MMSE_n_realization[i, k] = h_i.conj().T @ w
        
        interf_P_MMSE_vec += np.sum(np.abs(interf_P_MMSE_n_realization)**2, axis=1) / nbrOfRealizations

    SE_P_MMSE = prelogFactor * np.log2(1 + (np.abs(signal_P_MMSE)**2) / (interf_P_MMSE_vec - np.abs(signal_P_MMSE)**2 + 1))
    
    return SE_MR, SE_P_MMSE

# =============================================================================
# PHẦN 2: TRIỂN KHAI THUẬT TOÁN PSO (THAY THẾ GA)
# =============================================================================

class PSO_Optimizer:
    def __init__(self, n_particles, n_dimensions, max_iter, rho_tot, variant="inertia"):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions # Số lượng UE (K)
        self.max_iter = max_iter
        self.rho_tot = rho_tot
        self.variant = variant # "original" hoặc "inertia"
        
        # Tham số PSO (Tham khảo mục 1.3 và 4.2 trong PSO.pdf)
        self.c1 = 1.5 # Hệ số nhận thức
        self.c2 = 1.5 # Hệ số xã hội
        
        # Inertia Weight (w) - Mục 4.2 trong PDF
        self.w_max = 0.9
        self.w_min = 0.4
        self.w = self.w_max # Giá trị khởi tạo
        
        # Giới hạn vận tốc (Velocity Clamping - Mục 4.3 trong PDF)
        self.v_max = 0.2 * (rho_tot) 
        self.v_min = -self.v_max

    def optimize(self, fitness_func):
        # 1. Khởi tạo (Algorithm 1 trong PDF)
        # Vị trí: Công suất cho K users
        particles_x = np.random.rand(self.n_particles, self.n_dimensions)
        
        # Chuẩn hóa công suất ban đầu sao cho tổng công suất hợp lệ
        for i in range(self.n_particles):
            if np.mean(particles_x[i, :]) > 0:
                particles_x[i, :] = particles_x[i, :] * (self.rho_tot / np.mean(particles_x[i, :]))
        
        # Vận tốc khởi tạo
        particles_v = np.random.uniform(-1, 1, (self.n_particles, self.n_dimensions))
        
        # P-Best (Cá nhân) và G-Best (Toàn cục)
        pbest_x = particles_x.copy()
        pbest_fitness = -np.inf * np.ones(self.n_particles)
        
        gbest_x = np.zeros(self.n_dimensions)
        gbest_fitness = -np.inf
        
        history = []

        # 2. Vòng lặp chính (Algorithm 2 trong PDF)
        for it in range(self.max_iter):
            # Cập nhật Inertia Weight (Giảm tuyến tính theo thời gian - Biến thể phổ biến)
            if self.variant == "inertia":
                self.w = self.w_max - (self.w_max - self.w_min) * it / self.max_iter
            else:
                self.w = 1.0 # Original PSO không có w (hoặc w=1)

            # Đánh giá Fitness
            for i in range(self.n_particles):
                # Ràng buộc: Công suất không âm
                current_rho = np.abs(particles_x[i, :])
                # Chuẩn hóa lại tổng công suất (Constraint handling)
                if np.mean(current_rho) > 0:
                    current_rho = current_rho * (self.rho_tot / np.mean(current_rho))
                
                # Gọi hàm tính SE
                fitness_val = fitness_func(current_rho)
                
                # Cập nhật P-Best (Mục 1.4 bước 3 trong PDF)
                if fitness_val > pbest_fitness[i]:
                    pbest_fitness[i] = fitness_val
                    pbest_x[i, :] = current_rho.copy()
                    
                # Cập nhật G-Best (Mục 1.4 bước 4 trong PDF)
                if fitness_val > gbest_fitness:
                    gbest_fitness = fitness_val
                    gbest_x = current_rho.copy()
            
            # Cập nhật Vận tốc và Vị trí (Mục 1.3 trong PDF)
            for i in range(self.n_particles):
                r1 = np.random.rand(self.n_dimensions)
                r2 = np.random.rand(self.n_dimensions)
                
                # Phương trình (3) trong PDF (có w) hoặc (1) nếu w=1
                particles_v[i, :] = (self.w * particles_v[i, :] + 
                                     self.c1 * r1 * (pbest_x[i, :] - particles_x[i, :]) + 
                                     self.c2 * r2 * (gbest_x - particles_x[i, :]))
                
                # Velocity Clamping (Mục 4.3)
                particles_v[i, :] = np.clip(particles_v[i, :], self.v_min, self.v_max)
                
                # Cập nhật vị trí: Phương trình (2) trong PDF
                particles_x[i, :] = particles_x[i, :] + particles_v[i, :]

            print(f"PSO Iter {it+1}/{self.max_iter}: Best Min SE = {gbest_fitness:.4f}")
            history.append(gbest_fitness)
            
        return gbest_x, history

# Hàm wrapper để gọi PSO
def run_pso_power_allocation(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, R, pilotIndex, rho_tot, variant="inertia"):
    
    # Định nghĩa hàm Fitness: Đầu vào là vector rho, đầu ra là Min SE
    def fitness_function(rho_vector):
        # Lưu ý: Giảm số lượng realization xuống 5 trong quá trình tối ưu để chạy nhanh hơn (như trong ga_scf.py)
        _, SE_P = compute_se_downlink_fast(Hhat, H, D, B, C, tau_c, tau_p, 5, N, K, L, p, np.zeros((L,K)), R, pilotIndex, rho_vector)
        return np.min(SE_P) # Max-Min Fairness

    # Cấu hình PSO
    pso = PSO_Optimizer(n_particles=15, n_dimensions=K, max_iter=10, rho_tot=rho_tot, variant=variant)
    
    print(f"--- Running PSO ({variant}) ---")
    best_rho, _ = pso.optimize(fitness_function)
    return best_rho

# =============================================================================
# PHẦN 3: MAIN SIMULATION (ĐÃ SỬA LỖI)
# =============================================================================
if __name__ == "__main__":
    selectSimulationSetup = 2 
    
    if selectSimulationSetup == 1:
        L, N = 400, 1
    elif selectSimulationSetup == 2:
        L, N = 100, 4

    K = 20 # Giảm số user để test nhanh (gốc là 100)
    tau_c = 200
    tau_p = 10
    p = 100
    rho_tot = 1000 # mW

    nbrOfSetups = 5 # Số lượng setup mạng
    nbrOfRealizations = 100 # Số lượng mẫu kênh

    SE_MR_tot = []
    SE_P_MMSE_Basic_tot = []
    SE_P_MMSE_PSO_Orig_tot = []
    SE_P_MMSE_PSO_Var_tot = []

    print(f"Starting simulation with L={L}, N={N}, K={K}...")
    
    for n in range(nbrOfSetups):
        print(f"\n=== Setup {n+1} out of {nbrOfSetups} ===")
        
        gainOverNoisedB, R, pilotIndex, D = generateSetup(L, K, N, tau_p)
        Hhat, H, B, C = functionChannelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)
        
        # 1. Heuristic / Equal Power (Baseline)
        rho_central_heuristic = (rho_tot/tau_p)*np.ones(K)
        
        # Tạo rho_dist giả (Dummy) để khớp tham số hàm (chỉ dùng cho tính SE_MR bên trong hàm)
        rho_dist = np.zeros((L, K)) 
        
        # Tính Baseline
        SE_MR, SE_P_MMSE = compute_se_downlink_fast(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho_dist, R, pilotIndex, rho_central_heuristic)
        SE_MR_tot.extend(SE_MR)
        SE_P_MMSE_Basic_tot.extend(SE_P_MMSE)
        
        # 2. PSO Original (w = 1 constant)
        best_rho_pso_orig = run_pso_power_allocation(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, R, pilotIndex, rho_tot, variant="original")
        
        # --- SỬA LỖI Ở ĐÂY: Thêm rho_dist vào hàm gọi ---
        _, SE_PSO_Orig = compute_se_downlink_fast(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho_dist, R, pilotIndex, best_rho_pso_orig)
        SE_P_MMSE_PSO_Orig_tot.extend(SE_PSO_Orig)

        # 3. PSO Variant (Inertia Weight decaying)
        best_rho_pso_var = run_pso_power_allocation(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, R, pilotIndex, rho_tot, variant="inertia")
        
        # --- SỬA LỖI Ở ĐÂY: Thêm rho_dist vào hàm gọi ---
        _, SE_PSO_Var = compute_se_downlink_fast(Hhat, H, D, B, C, tau_c, tau_p, nbrOfRealizations, N, K, L, p, rho_dist, R, pilotIndex, best_rho_pso_var)
        SE_P_MMSE_PSO_Var_tot.extend(SE_PSO_Var)

    # Vẽ biểu đồ CDF
    plt.figure(figsize=(10, 6))
    def plot_cdf(data, style, label):
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, yvals, style, linewidth=2, label=label)

    plot_cdf(SE_P_MMSE_PSO_Var_tot, 'r-', 'P-MMSE (PSO Variant - Inertia)')
    plot_cdf(SE_P_MMSE_PSO_Orig_tot, 'g--', 'P-MMSE (PSO Original)')
    plot_cdf(SE_P_MMSE_Basic_tot, 'b-.', 'P-MMSE (Equal Power)')
    plot_cdf(SE_MR_tot, 'k:', 'MR (Reference)')

    plt.xlabel('Spectral efficiency [bit/s/Hz]')
    plt.ylabel('CDF')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('CDF Comparison: PSO Variants vs Baseline in Scalable Cell-Free')
    plt.show()