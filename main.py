import numpy as np
import matplotlib.pyplot as plt
import time

class EvolutionStrategy:
    def __init__(self, interval=(-5.12, 5.12), dim=2, sigma=0.5, max_iter=100, show_intermediate=False):
        self.interval = interval  # Границы поиска
        self.dim = dim  # Размерность задачи
        self.sigma = sigma  # Стандартное отклонение для мутации
        self.max_iter = max_iter  # Максимальное количество итераций
        self.show_intermediate = show_intermediate  # Флаг для промежуточного вывода

        # Инициализация начальной особи случайным образом в пределах границ
        self.current_solution = np.random.uniform(self.interval[0], self.interval[1], self.dim)
        self.best_solution = self.current_solution
        self.best_score = self.hyper_ellipsoid_function(self.current_solution)
        self.population_points = []  # Для сохранения точек популяции

    def hyper_ellipsoid_function(self, x):
        """ Функция гиперэллипсоида для многомерного случая """
        return sum(5 * (i + 1) * (x[i] ** 2) for i in range(len(x)))

    def mutate(self, parent):
        """ Оператор мутации """
        return parent + np.random.normal(0, self.sigma, size=self.dim)

    def optimize(self):
        """ Основной метод оптимизации """
        start_time = time.time()  # Начало измерения времени

        for iteration in range(self.max_iter):
            # Создание потомка
            offspring = self.mutate(self.current_solution)
            # Принудительное соблюдение границ
            offspring = np.clip(offspring, self.interval[0], self.interval[1])

            # Оценка приспособленности потомка
            offspring_score = self.hyper_ellipsoid_function(offspring)

            # Сохранение текущей точки популяции
            self.population_points.append(self.current_solution.copy())

            # Если потомок лучше родителя, заменяем родителя
            if offspring_score < self.best_score:
                self.current_solution = offspring
                self.best_score = offspring_score
                self.best_solution = offspring

            # Вывод текущего состояния каждые 10 итераций, если включен промежуточный вывод
            if self.show_intermediate and iteration % 10 == 0:
                print(f"Итерация {iteration}: лучшее значение = {self.best_score}")

        end_time = time.time()  # Окончание измерения времени
        execution_time = end_time - start_time
        return self.best_solution, self.best_score, execution_time

    def draw_3d_plot(self, best_solution=None, highlight_best=False, generation=None):
        """ Построение 3D графика с отображением функции и текущих решений """
        if self.dim != 2:
            print("Визуализация доступна только для n=2.")
            return

        x1 = np.linspace(self.interval[0], self.interval[1], 100)
        x2 = np.linspace(self.interval[0], self.interval[1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        Z = self.hyper_ellipsoid_function([X1, X2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

        # Отображение точек популяции
        population_x = [p[0] for p in self.population_points]
        population_y = [p[1] for p in self.population_points]
        population_z = [self.hyper_ellipsoid_function(p) for p in self.population_points]
        ax.scatter(population_x, population_y, population_z, color='blue', s=20, label='Точки популяции')

        # Отображение текущего решения
        if highlight_best and best_solution is not None:
            best_fitness = self.hyper_ellipsoid_function(best_solution)
            ax.scatter(best_solution[0], best_solution[1], best_fitness, color='red', s=100, label='Лучшее решение', zorder=2)

        # Добавляем номер поколения на график
        if generation is not None:
            ax.text2D(0.05, 0.95, f"Поколение: {generation}", transform=ax.transAxes)

        ax.set_title('Гиперэллипсоидная функция с точками популяции')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1, x2)')
        ax.legend()
        plt.show()


# Основной блок выполнения экспериментов
def run_experiments():
    experiments = [
        {"dim": 2, "sigma": 0.5, "max_iter": 100},
        {"dim": 2, "sigma": 1.0, "max_iter": 100},
        {"dim": 2, "sigma": 0.5, "max_iter": 200},
        {"dim": 3, "sigma": 0.5, "max_iter": 100},
        {"dim": 3, "sigma": 1.0, "max_iter": 100},
        {"dim": 3, "sigma": 0.5, "max_iter": 200}
    ]

    for i, params in enumerate(experiments, 1):
        print(f"\nВызов {i}: Параметры: dim={params['dim']}, sigma={params['sigma']}, max_iter={params['max_iter']}")
        es = EvolutionStrategy(interval=(-5.12, 5.12), dim=params['dim'], sigma=params['sigma'],
                               max_iter=params['max_iter'], show_intermediate=False)
        best_solution, best_score, exec_time = es.optimize()
        print(f"Лучшее решение: {best_solution}, значение функции: {best_score}, время выполнения: {exec_time:.4f} секунд")


if __name__ == '__main__':
    # Вызов с графиком для n=2
    print("\nОсновной вызов с графиком: dim=2, sigma=0.5, max_iter=100")
    es = EvolutionStrategy(interval=(-5.12, 5.12), dim=2, sigma=0.5, max_iter=100, show_intermediate=True)
    best_solution, best_score, exec_time = es.optimize()

    # Вывод результатов
    print(f"Найденное решение: {best_solution}, значение функции: {best_score}, время выполнения: {exec_time:.4f} секунд")

    # Визуализация результатов
    es.draw_3d_plot(best_solution=best_solution, highlight_best=True, generation=es.max_iter)

    # Запуск экспериментов без графиков
    run_experiments()
