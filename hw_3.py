r"""
Динамика: c_i(t+1) = δ c_i(t) − α y_i(t) − ∑_{j∈N\{i}} [β g_ij g_ji + γ(1 − g_ij g_ji)] y_j(t)
Прибыль: J_i = ∑_{t=0}^{T-1} ρ^t [(p − c_i(t) − ∑_{j=1}^{n} u_j(t)) u_i(t) − (ε/2)y_i²(t) − ∑_{j∈N\{i}} π g_ij g_ji]
                  + ∑_{j=1}^{n} φ_ij * [δ c_j(t) − α y_j(t) − ∑_{k∈N\{j}} [β g_jk g_jk + γ(1 − g_jk g_jk)] y_k(t)]
Инвестиционные усилия y: y_i(t) = (−α / (ρ^t ε)) φ_ii(t+1)
Объем производства u: u_i(t) = (p − (n+1)c_i(t) + ∑_{j∈N\{i}} c_j(t)) / (n+1)
Рекуррентность φ: φ_ii(t) = −ρ^t u_i(t) + δ φ_ii(t+1)
Доп: φ_ii(T) = −ρ^t η, φ_ij(T) = 0 для i ≠ j
       φ_ij(t) = 0  для i ≠ j
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def setting_values(p, n, ro, T, delta, nu_i, nu, epsilon, alpha, beta, gamma, c0, pi):
    if n < 2:
        raise ValueError("n >= 2")
    if not (0 < ro <= 1):
        raise ValueError(f"ρ ∈ (0, 1], получено {ro}")
    if T < 1:
        raise ValueError(f"T >= 1, получено {T}")
    if delta < 1:
        raise ValueError(f"δ >= 1, получено {delta}")
    if epsilon <= 0:
        raise ValueError(f"ε > 0, получено {epsilon}")
    if not (0 <= gamma <= beta <= alpha):
        raise ValueError(f"0 ≤ γ ≤ β ≤ α")
    if len(c0) != n:
        raise ValueError(f"|c0| = {n}")
    if len(nu_i) != n:
        raise ValueError(f"|η_i| = {n}")
    if not isinstance(pi, np.ndarray):
        pi = np.array(pi, dtype=float)
    if pi.shape != (n, n):
        raise ValueError(f"|π| = {n}×{n}, получено {pi.shape}")
    u = np.zeros((n, T))  # u_i(t)
    y = np.zeros((n, T))  # y_i(t)
    g = np.zeros((n, n, T))  # g_ij(t)
    c = np.zeros((n, T + 1))  # c_i(t)
    c[:, 0] = c0
    phi = np.zeros((n, n, T + 1))
    print(f"n = {n} фирм, T = {T} периодов")
    print(f"c(0) = {c0}")
    print(f"Параметры: p={p}, ρ={ro}, δ={delta}, ε={epsilon}")
    print(f"α={alpha}, β={beta}, γ={gamma}, ν={nu}, ν_i={nu_i}")

    return u, y, g, c, phi, pi


def cost_dynamics(c, y, g, delta, alpha, beta, gamma):
    r"""
    c_i(t+1) = δ c_i(t) − α y_i(t) − ∑_{j∈N\{i}} [β g_ij g_ji + γ(1 − g_ij g_ji)] y_j(t)
    """
    n, T_plus_1 = c.shape
    T = T_plus_1 - 1
    for t in range(T):
        for i in range(n):
            own_effect = alpha * y[i, t]
            neighbor_effect = 0.0
            for j in range(n):
                if j != i:
                    cooperation = g[i, j, t] * g[j, i, t]
                    coeff = beta * cooperation + gamma * (1 - cooperation)
                    neighbor_effect += coeff * y[j, t]
            # c[i, t + 1] = max(delta * c[i, t] - own_effect - neighbor_effect, 0)
            c[i, t + 1] = delta * c[i, t] - own_effect - neighbor_effect
    return c


def calculate_phi(phi, c, u, ro, T, delta, n, nu, nu_i):
    r"""
    φ: φ_ii(t) = −ρ^t u_i(t) + δ φ_ii(t+1)
    Доп: φ_ii(T) = −ρ^t η, φ_ij(T) = 0 для i ≠ j
         φ_ij(t) = 0  для i ≠ j
    """
    for i in range(n):
        phi[i, i, T] = -(ro ** T) * nu
        for j in range(n):
            if j != i:
                phi[i, j, T] = 0

    # обратная рекуррентность: от t = T-1 к t = 0
    for t in range(T - 1, -1, -1):
        for i in range(n):
            # φ_ii(t) = −ρ^t u_i(t) + δ φ_ii(t+1)
            phi[i, i, t] = -(ro ** t) * u[i, t] + delta * phi[i, i, t + 1]
            # φ_ij(t) = 0  для j ≠ i
            for j in range(n):
                if j != i:
                    phi[i, j, t] = 0

    return phi


def calculate_cooperation(g, pi, beta, gamma, y, phi, ro, t, n):
    r"""
    Матрица кооперации
    pi_ij < (-1 / (ρ^t)) * (β - γ) * y_j(t) * φ_ii(t)
    """
    g_new = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if g[i, j] == 1 and g[j, i] == 1:
                benefit_i = (beta - gamma) * y[j, t] * (-phi[i, i, t + 1])
                net_i = benefit_i - pi[i, j]
                benefit_j = (beta - gamma) * y[i, t] * (-phi[j, j, t + 1])
                net_j = benefit_j - pi[j, i]
                if net_i > 0 and net_j > 0:
                    g_new[i, j] = 1
                    g_new[j, i] = 1

    return g_new


def calculate_volume_of_production(p, n, c, t):
    r"""
    u_i(t) = (p − (n+1)c_i(t) + ∑_{j∈N\{i}} c_j(t)) / (n+1)
    """
    u_t = np.zeros(n)
    for i in range(n):
        sum_c_j_all = np.sum(c[:, t])  # ∑_{j=1}^{n} c_j(t)
        sum_c_j_except_i = sum_c_j_all - c[i, t]  # ∑_{j∈N\{i}} c_j(t)
        u_t[i] = (p - (n + 1) * c[i, t] + sum_c_j_except_i) / (n + 1)
        u_t[i] = max(u_t[i], 0)

    return u_t


def calculate_invest(alpha, ro, epsilon, phi, t, n):
    r"""
    y_i(t) = (−α / (ρ^t ε)) φ_ii(t+1)
    Если φ_ii(t+1) >= 0, то y_i(t) <= 0, что нарушает ограничение y_i(t) >= 0
    Тогда y_i(t) = 0
    """
    y_t = np.zeros(n)
    for i in range(n):
        if ro ** t > 1e-9:
            y_t[i] = (-alpha / ((ro ** t) * epsilon)) * phi[i, i, t + 1]
        y_t[i] = max(y_t[i], 0)

    return y_t


def profit_i(i, u, y, g, c, p, ro, epsilon, pi, nu_i, nu, phi, alpha, beta, gamma, delta, n, T):
    r"""
    J_i = ∑_{t=0}^{T-1} ρ^t [(p − c_i(t) − ∑_{j=1}^{n} u_j(t)) u_i(t) − (ε/2)y_i²(t) − ∑_{j∈N\{i}} π g_ij g_ji]
                  + ∑_{j=1}^{n} φ_ij * [δ c_j(t) − α y_j(t) − ∑_{k∈N\{j}} [β g_jk g_jk + γ(1 − g_jk g_jk)] y_k(t)]
    """
    J = 0.0
    for t in range(T):
        # (p − c_i(t) − ∑_{j=1}^{n} u_j(t)) u_i(t)
        sum_u = np.sum(u[:, t])
        clear_profit = (p - c[i, t] - sum_u) * u[i, t]
        # − (ε/2) y_i²(t)
        investments = (epsilon / 2.0) * (y[i, t] ** 2)
        # − ∑_{j≠i} π_ij g_ij(t) g_ji(t)
        costs_of_cooperation = 0.0
        for j in range(n):
            if j != i:
                costs_of_cooperation += pi[i, j] * g[i, j, t] * g[j, i, t]
        # ∑_{j=1}^n φ_ij(t) * [δ c_j(t) − α y_j(t) − ∑_k [β g_jk g_kj + γ(1 − g_jk g_kj)] y_k(t)]
        stage_profit = clear_profit - investments - costs_of_cooperation
        J += (ro ** t) * stage_profit
    return J


def solve_game_forward(p, n, ro, T, delta, nu_i, nu, epsilon, alpha, beta, gamma, c0, pi, g):
    u, y, g_calc, c, phi, pi = setting_values(
        p, n, ro, T, delta, nu_i, nu, epsilon, alpha, beta, gamma, c0, pi)
    g_calc = g.copy()
    max_iterations = 10
    tolerance = 1e-2
    for iteration in range(max_iterations):
        calculate_phi(phi, c, u, ro, T, delta, n, nu, nu_i)
        y_old = y.copy()
        for t in range(T):
            y[:, t] = calculate_invest(alpha, ro, epsilon, phi, t, n)
        for t in range(T):
            u[:, t] = calculate_volume_of_production(p, n, c, t)
        for t in range(1, T):
            g_calc[:, :, t] = calculate_cooperation(
                g_calc[:, :, t - 1],
                pi, beta, gamma,
                y, phi, ro, t - 1, n)
        cost_dynamics(c, y, g_calc, delta, alpha, beta, gamma)
        delta_y = np.max(np.abs(y - y_old))
        print(f"Итерация {iteration + 1}: max|Δy| = {delta_y:.2e}")
        if delta_y < tolerance:
            print(f"Сходимость достигнута за {iteration + 1} итераций")
    c = np.zeros((n, T + 1))
    c[:, 0] = c0
    cost_dynamics(c, y, g_calc, delta, alpha, beta, gamma)
    phi = np.zeros((n, n, T + 1))
    calculate_phi(phi, c, u, ro, T, delta, n, nu, nu_i)
    profits = np.zeros(n)
    for i in range(n):
        profits[i] = profit_i(i, u, y, g_calc, c, p, ro, epsilon, pi, nu_i, nu,
                                phi, alpha, beta, gamma, delta, n, T)
    return u, y, c, profits, g_calc


def print_results(n, T, u, y, c, profits):
    print("Инвестиционные усилия y_i(t)")
    print(f"{'Фирма':<10} | " + " | ".join([f"t={t}" for t in range(T)]))
    for i in range(n):
        row = f"y_{i + 1}(t)   "
        for t in range(T):
            row += f" | {y[i, t]:6.4f}"
        print(row)
    print("Объем товара u_i(t)")
    print(f"{'Фирма':<10} | " + " | ".join([f"t={t}" for t in range(T)]))
    for i in range(n):
        row = f"u_{i + 1}(t)   "
        for t in range(T):
            row += f" | {u[i, t]:6.4f}"
        print(row)
    print("Издержки c_i(t)")
    print(f"{'Фирма':<10} | " + " | ".join([f"t={t}" for t in range(T + 1)]))
    for i in range(n):
        row = f"c_{i + 1}(t)   "
        for t in range(T + 1):
            row += f" | {c[i, t]:6.4f}"
        print(row)
    print("Прибыль")
    for i in range(n):
        print(f"J_{i + 1} = {profits[i]:12.6f}")
    print(f"Итого: {np.sum(profits):12.6f}")


def print_cooperation_matrices(g, n, T):
    r"""
    Выводит матрицы кооперации g_ij(t) для каждого периода
    g[i,j,t] = 1: фирма i предлагает кооперацию фирме j в период t
    """
    print("Матрицы кооперации g_ij(t)")
    print()

    for t in range(T):
        print(f"\nПериод t={t}:")
        print()
        for j in range(n):
            print(f"F{j + 1}  ", end="")
        print()
        for i in range(n):
            print(f"F{i + 1}: ", end="")
            for j in range(n):
                val = int(g[i, j, t])
                print(f" {val}  ", end="")
            print()
        print("\nВзаимные кооперации:")
        mutual_found = False
        for i in range(n):
            for j in range(i + 1, n):
                if g[i, j, t] == 1 and g[j, i, t] == 1:
                    mutual_found = True
                    print(f"F{i + 1} ↔ F{j + 1}")
        if not mutual_found:
            print("  (нет взаимных кооперациий)")


def plot_cooperation_networks(g, n, T):
    """
    Визуализация графов для каждого периода t
    """
    fig, axes = plt.subplots(1, T, figsize=(7 * T, 6))
    if T == 1:
        axes = [axes]
    for t in range(T):
        ax = axes[t]
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if i != j and g[i, j, t] == 1:
                    G.add_edge(i, j)
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos,
                               node_color='lightblue',
                               node_size=1500,
                               ax=ax,
                               edgecolors='navy',
                               linewidths=2)
        labels = {i: f'F{i + 1}' for i in range(n)}
        nx.draw_networkx_labels(G, pos,
                                labels,
                                font_size=13,
                                ax=ax)
        nx.draw_networkx_edges(G, pos,
                               edge_color='darkgray',
                               arrows=True,
                               arrowsize=25,
                               arrowstyle='-|>',
                               width=2.5,
                               ax=ax,
                               connectionstyle="arc3,rad=0.1",
                               alpha=0.7)
        ax.set_title(f'Граф сотрудничества (период t={t})',
                     fontsize=14, weight='bold', pad=20)
        ax.axis('off')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    p = 500
    n = 4
    ro = 0.95
    T = 3
    delta = 1.07
    nu_i = np.array([100000, 100000, 100000, 100000])
    nu = 1000
    epsilon = 1000
    alpha = 1.8
    beta = 1
    gamma = 0.5
    c0 = np.array([100, 100, 100, 100])
    pi = np.array([[0, 800, 800, 800], [800, 0, 800, 800], [900, 900, 0, 900], [1100, 1100, 1100, 0]])
    g = np.ones((n, n, T))
    for t in range(T):
        np.fill_diagonal(g[:, :, t], 0)
    u_opt, y_opt, c_opt, profits, g_opt = solve_game_forward(p, n, ro, T, delta, nu_i, nu, epsilon, alpha, beta, gamma, c0, pi, g)
    print_results(n, T, u_opt, y_opt, c_opt, profits)
    print_cooperation_matrices(g_opt, n, T)
    fig = plot_cooperation_networks(g_opt, n, T)
    fig.savefig("cooperation_networks.png", dpi=300, bbox_inches="tight")