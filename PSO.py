import math
import random
import numpy
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor


grid_max = 50
grid_min = -50

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", np.ndarray, fitness=creator.FitnessMin, speed=None, smin=None, smax=None, best=None)
creator.create("BestGP", np.ndarray, fitness=creator.FitnessMin)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle([random.uniform(pmin, pmax) for _ in range(size)])
    part.speed = np.array([random.uniform(smin, smax) for _ in range(size)])
    part.smin = smin
    part.smax = smax

    return part

def gp_generate(index_x, index_y):
    best_gp1 = [index_x, index_y]
    best_gp = np.array(best_gp1)

    return best_gp

def updateParticle(part, best, phi1, phi2):
    u1 = np.array([random.uniform(0, phi1) for _ in range(len(part))])
    u2 = np.array([random.uniform(0, phi2) for _ in range(len(part))])
    v_u1 = u1 * (part.best - part)
    v_u2 = u2 * (best - part)
    part.speed = v_u1 + v_u2 + part.speed
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = part + part.speed

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2, pmin=grid_min, pmax=grid_max, smin=-2, smax=2)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2, phi2=2)
toolbox.register("evaluate", benchmarks.bohachevsky)

def plot_evolucion(log):
    gen = log.select("gen")
    fit_mins = log.select("min")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")

    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    ax1.plot(gen, fit_maxs, "r")
    ax1.plot(gen, fit_ave, "--k")
    ax1.fill_between(gen, fit_mins, fit_maxs,
                     where=fit_maxs >= fit_mins,
                     facecolor="g", alpha=0.2)
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Fitness")
    ax1.legend(["Min", "Max", "Avg"])
    plt.grid(True)

def plot_movimiento(x_a, y_a):
    plt.figure(2)
    plt.scatter(x_a, y_a)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.title("Movimiento de las partículas")
    plt.show()
    plt.close()

sigma_kernel = 0.7
ker = RBF(length_scale=0.6)

f_max = grid_max
f_min = grid_min
j = 0
i = 0
map, map2 = np.mgrid[f_min:f_max:1, f_min:f_max:1]

dimx = map2.shape[0]
dimy = map2.shape[1]

x_grid = []
y_grid = []

x = np.arange(grid_min, grid_max, 1)
y = np.arange(grid_min, grid_max, 1)
for i in range(len(y)):
    for j in range(len(x)):
        x_grid.append(x[j])
        y_grid.append(y[i])

X_grid = np.array(x_grid).reshape(-1, 1)
X_grid_shape = X_grid.shape
x_zero = X_grid_shape[0]
Y_grid = np.array(y_grid).reshape(-1, 1)
Y_grid_shape = Y_grid.shape

X_test = np.concatenate([X_grid, Y_grid], axis=1).reshape(-1, 2)
X_zero = np.zeros((x_zero, 2))

res = ker(X_test)

noise = 1
gpr = GaussianProcessRegressor(kernel=ker, alpha=noise**2)

def gaussian_regression(x_train, y_train):
    gpr.fit(x_train, y_train)
    gpr.get_params()

    mu, sigma = gpr.predict(X_test, return_std=True)

    Z_var = sigma.reshape(dimx, dimy)
    Z_mean = mu.reshape(dimx, dimy)

    return sigma, mu, Z_var, Z_mean


def plot_gaussian(x_a, y_a, n, Z_var, Z_mean):
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 6))

    im1 = axs[0].scatter(x_ga, y_ga, c=n, cmap="gist_rainbow", marker='.')

    im2 = axs[0].imshow(Z_var, interpolation='bilinear', origin='lower', cmap="viridis")
    plt.colorbar(im2, ax=axs[0], format='%.2f', label='σ', shrink=1)
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    axs[0].set_aspect('equal')
    axs[0].grid(True)

    im3 = axs[1].imshow(Z_mean, interpolation='bilinear', origin='lower', cmap="jet")
    plt.colorbar(im3, ax=axs[1], format='%.2f', label='µ', shrink=1)
    axs[1].set_xlabel("x [m]")
    axs[1].set_ylabel("y [m]")
    axs[1].set_aspect('equal')
    axs[1].grid(True)

    plt.show()

def data(x_p, y_p, y_data):
    x_a = numpy.array(x_p).reshape(-1, 1)
    y_a = numpy.array(y_p).reshape(-1, 1)
    x_train = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
    y_train = numpy.array(y_data).reshape(-1, 1)

    return x_a, y_a, x_train, y_train

def sigmamax(Z_var):
    sigma_max = np.max(Z_var)
    print(sigma_max)
    index_sigma = np.where(Z_var == sigma_max)
    print(index_sigma)
    index_x1 = index_sigma[1]
    index_x2 = index_x1[0]
    index_y1 = index_sigma[0]
    index_y2 = index_y1[0]
    index_x = float(index_x2)
    index_y = float(index_y2)
    return sigma_max, index_x, index_y

bound_min = grid_min + 1
bound_max = grid_max - 1

def limit(part):
    if part[0] <= grid_min:
        part[0] = bound_min
    if part[0] >= grid_max:
        part[0] = bound_max
    if part[1] <= grid_min:
        part[1] = bound_min
    if part[1] >= grid_max:
        part[1] = bound_max
    return part

total_suma = 0
suma_data = []
nsamples = 0
i = 0

def mse(y_data, mu_data, bench_max, samples):
    total_suma = 0
    y_array = np.array(y_data)
    mu_array = np.array(mu_data)
    for i in range(len(mu_array)):
        total_suma = (float(y_array[i]) - float(mu_array[i])) ** 2 + total_suma
    MSE1 = total_suma / samples
    return MSE1

part1x = []
part1y = []
part2x = []
part2y = []
part3x = []
part3y = []
part4x = []
part4y = []

def distance(g, n_data, part, dist1, dist2, dist3, dist4):
    b = g - 1
    if n_data == 1.0:
        part1x.append(part[0])
        part1y.append(part[1])
        dist1 = math.sqrt((part1x[g] - part1x[b]) ** 2 + (part1y[g] - part1y[b]) ** 2) + dist1
    elif n_data == 2.0:
        part2x.append(part[0])
        part2y.append(part[1])
        dist2 = math.sqrt((part2x[g] - part2x[b]) ** 2 + (part2y[g] - part2y[b]) ** 2) + dist2
    elif n_data == 3.0:
        part3x.append(part[0])
        part3y.append(part[1])
        dist3 = math.sqrt((part3x[g] - part3x[b]) ** 2 + (part3y[g] - part3y[b]) ** 2) + dist3
    elif n_data == 4.0:
        part4x.append(part[0])
        part4y.append(part[1])
        dist4 = math.sqrt((part4x[g] - part4x[b]) ** 2 + (part4y[g] - part4y[b]) ** 2) + dist4
    return dist1, dist2, dist3, dist4

e1 = str('PSOError26.xlsx')
e2 = str('PSOMU26.xlsx')
e3 = str('PSOSigma26.xlsx')
e4 = str('PSODist26.xlsx')

random.seed(26)
pop = toolbox.population(n=4)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

logbook = tools.Logbook()
logbook.header = ["gen", "evals"] + stats.fields

lista_best = list()
lista_gp_best = list()


best = pop[0]
n_data = float(1)
x_p = []
y_p = []
array = []
y_data = []
x_train = []
n = []
benchmark_data = []

for i in range(len(X_test)):
    benchmark_value = toolbox.evaluate(X_test[i])
    benchmark_data.append(benchmark_value)
benchmark_array = np.array(benchmark_data)
meanz = np.nanmean(benchmark_array)
stdz = np.nanstd(benchmark_array)
benchmark_array = (benchmark_array - meanz) / stdz
bench_min = min(benchmark_array)
bench_max = abs(bench_min[0])
Benchmark_plot = benchmark_array.reshape(dimx, dimy)

bench_data = []
MSE_data = []
MSE2_data = []
itMSE_data = []
it = []
mu_data = []
sigma_data = []
part_int_data = []
samples = 0
g = 0
x_g = []
y_g = []

for part in pop:
    x_p.append(part[0])
    y_p.append(part[1])
    x_bench = int(part[0])
    y_bench = int(part[1])
    x_gap = int(part[0]) + abs(grid_min)
    y_gap = int(part[1]) + + abs(grid_min)
    x_g.append(x_gap)
    y_g.append(y_gap)
    part.fitness.values = [Benchmark_plot[x_bench + abs(grid_min)][y_bench + abs(grid_min)]]
    part_int_fitness = [Benchmark_plot[x_bench + abs(grid_min)][y_bench + abs(grid_min)]]
    part_int_data.append(part_int_fitness)
    y_data.append(part.fitness.values)
    n.append(n_data)
    part.best = creator.Particle(part)
    part.best.fitness.values = part.fitness.values
    if best.fitness < part.fitness:
        best = creator.Particle(part)
        best.fitness.values = part.fitness.values
    if n_data == 1:
        part1x.append(part[0])
        part1y.append(part[1])
    if n_data == 2:
        part2x.append(part[0])
        part2y.append(part[1])
    if n_data == 3:
        part3x.append(part[0])
        part3y.append(part[1])
    if n_data == 4:
        part4x.append(part[0])
        part4y.append(part[1])
    n_data += float(1)
    x_a, y_a, x_train, y_train = data(x_p, y_p, y_data)
    sigma, mu, Z_var, Z_mean = gaussian_regression(x_train, y_train)
    mu_value = Z_mean[x_bench + abs(grid_min)][y_bench + abs(grid_min)]
    sigma_value = Z_var[x_bench + abs(grid_min)][y_bench + abs(grid_min)]
    sigma_data.append(sigma_value)
    mu_data.append(mu_value)
    samples += 1

MSE = mse(part_int_data, mu_data, bench_max, samples)
MSE_data.append(MSE)
it.append(g)

print("PFV", part.fitness.values)

lista_best.append(best)

for part in pop:
    toolbox.update(part, best)

k = 0
GEN = 50
sigma_max_ant = 0
sigma_revision = 0
sigma_ant = 0
suma_total = 0
phi1 = 2
phi2 = 2
phi3 = 0

dist1 = 0
dist2 = 0
dist3 = 0
dist4 = 0
n_data = float(1)
nd = float(1)


for g in range(GEN):
    for part in pop:
        part = limit(part)
        x_p.append(part[0])
        y_p.append(part[1])
        x_bench = int(part[0])
        y_bench = int(part[1])
        if g % 10 == 0:
            x_gap = int(part[0]) + abs(grid_min)
            y_gap = int(part[1]) + abs(grid_min)
            x_g.append(x_gap)
            y_g.append(y_gap)
            n.append(n_data)
            n_data += float(1)
            if n_data > 4:
                n_data = float(1)
        part.fitness.values = [Benchmark_plot[x_bench + abs(grid_min)][y_bench + abs(grid_min)]]
        part_int_fitness = [Benchmark_plot[x_bench + abs(grid_min)][y_bench + abs(grid_min)]]
        part_int_data.append(part_int_fitness)
        dist1, dist2, dist3, dist4 = distance(g, nd, part, dist1, dist2, dist3, dist4)
        nd += float(1)
        if nd > 4:
            nd = float(1)
        y_data.append(part.fitness.values)
        if part.best.fitness < part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        if best.fitness < part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values
            lista_best.append(best)
        x_a, y_a, x_train, y_train = data(x_p, y_p, y_data)
        sigma, mu, Z_var, Z_mean = gaussian_regression(x_train, y_train)
        mu_value = Z_mean[x_bench + abs(grid_min)][y_bench + abs(grid_min)]
        sigma_value = Z_var[x_bench + abs(grid_min)][y_bench + abs(grid_min)]
        sigma_data.append(sigma_value)
        mu_data.append(mu_value)
        samples += 1

    MSE = mse(y_data, mu_data, bench_max, samples)
    MSE_data.append(MSE)
    it.append(g)

    for part in pop:
        toolbox.update(part, best)

    logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
    print(logbook.stream)

x_a = numpy.array(x_p).reshape(-1, 1)
y_a = numpy.array(y_p).reshape(-1, 1)
x_ga = numpy.array(x_g).reshape(-1, 1)
y_ga = numpy.array(y_g).reshape(-1, 1)
part_total = np.concatenate([x_a, y_a], axis=1).reshape(-1, 2)
Z_zero = np.ones((dimx, dimy))
MSE_array = np.array(MSE_data)
MSE_max = max(MSE_array)
xlimit = MSE_max + MSE_max * 0.1
itMSE_array = np.array(itMSE_data)
prom = (dist1 + dist2 + dist3 + dist4) / 4
prom_data = []
prom_data.append(prom)

wb = openpyxl.Workbook()
hoja1 = wb.active
hoja1.append(MSE_data)
wb.save(e1)
hoja2 = wb.active
hoja2.append(mu_data)
wb.save(e2)
hoja3 = wb.active
hoja3.append(MSE_data)
wb.save(e3)
hoja4 = wb.active
hoja4.append(prom_data)
wb.save(e4)

plot_gaussian(x_a, y_a, n, Z_var, Z_mean)

plt.figure(2)
im4 = plt.imshow(Benchmark_plot, interpolation='bilinear', origin='lower', cmap="jet")
plt.colorbar(im4, format='%.2f', shrink=1)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid(True)
plt.show()

plt.figure(3)
plt.plot(it, MSE_data, '-')
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.grid(True)
plt.xlim([0, GEN])
plt.ylim([0, xlimit])
plt.title("Mean Square Error")
plt.show()