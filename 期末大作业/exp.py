import numpy as np

# 定义目标函数
def objective_function(x):
    return np.sum(x**2)

# 遗传算法参数
population_size = 100 # 种群大小
dimensions = 50 # 个体维度
generations = 5000 # 迭代次数
mutation_rate = 0.1 # 变异概率
x_min, x_max = -10, 10 # 取值范围

# 初始化种群（整数）
population = np.random.randint(x_min, x_max + 1, (population_size, dimensions))

# 适应度计算
def calculate_fitness(pop):
    return np.array([objective_function(ind) for ind in pop])

# 选择操作
def selection(pop, fitness):
    probabilities = 1 / (fitness + 1e-8)
    probabilities /= probabilities.sum()
    selected_indices = np.random.choice(len(pop), size=len(pop), p=probabilities)
    return pop[selected_indices]

# 交叉操作（保持整数）
def crossover(parent1, parent2):
    mask = np.random.rand(dimensions) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1.astype(int), child2.astype(int)

# 变异操作（保持整数）
def mutation(ind):
    if np.random.rand() < mutation_rate:
        mutate_idx = np.random.randint(0, dimensions)
        ind[mutate_idx] = np.random.randint(x_min, x_max + 1)
    return ind

# 遗传算法主循环
best_solution = None
best_fitness = float('inf')

convergence_counter = 0 # 收敛计数器
tolerance = 1e-6 # 允许的误差范围

for generation in range(generations):
    # 计算适应度
    fitness = calculate_fitness(population)
    
    # 更新全局最优
    current_best_idx = np.argmin(fitness)
    if fitness[current_best_idx] < best_fitness:
        best_solution = population[current_best_idx]
        best_fitness = fitness[current_best_idx]
    
    # 打印每代最优解
    if generation % 50 == 0 or generation == generations - 1:
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
    
    # 检查是否达到提前终止条件
    if best_fitness < tolerance:
        print(f"Algorithm converged at generation {generation}")
        break
    
    # 收敛判断：如果适应度变化小于阈值多代
    if generation > 0 and np.abs(prev_best_fitness - best_fitness) < tolerance:
        convergence_counter += 1
    else:
        convergence_counter = 0
    if convergence_counter > 500:
        print(f"Algorithm stopped due to convergence at generation {generation}")
        break

    prev_best_fitness = best_fitness  # 保存上一代最优值
    
    # 选择
    population = selection(population, fitness)
    
    # 交叉
    new_population = []
    for i in range(0, population_size, 2):
        parent1, parent2 = population[i], population[(i+1) % population_size]
        child1, child2 = crossover(parent1, parent2)
        new_population.append(child1)
        new_population.append(child2)
    population = np.array(new_population)
    
    # 变异
    population = np.array([mutation(ind) for ind in population])

# 输出结果
print(f"最优解：{best_solution}")
print(f"最小值：{best_fitness}")
