import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    m = len(matrix_a)         # Количество строк в A
    n = len(matrix_a[0])      # Количество столбцов в A
    p = len(matrix_b[0])      # Количество столбцов в B

    # Проверяем, можно ли перемножить матрицы
    if len(matrix_b) != n:
        raise ValueError("Число столбцов первой матрицы должно равняться числу строк второй матрицы.")

    # Создаем результатирующую матрицу C размером m x p
    C = [[0 for _ in range(p)] for _ in range(m)]

    # Умножаем матрицы
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return C


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # Парсим коэффициенты
    a1 = list(map(float, a_1.split()))
    a2 = list(map(float, a_2.split()))

    # Определяем функции F(x) и P(x)
    def F(x):
        return a1[0] * x ** 2 + a1[1] * x + a1[2]

    def P(x):
        return a2[0] * x ** 2 + a2[1] * x + a2[2]

    # Находим экстремумы F(x)
    res_F = minimize_scalar(F)
    extremum_F = res_F.x if res_F.success else None

    # Находим экстремумы P(x)
    res_P = minimize_scalar(P)
    extremum_P = res_P.x if res_P.success else None

    # Печатаем экстремумы
    if extremum_F is not None:
        print(f"Экстремум F(x): {extremum_F}, F({extremum_F}) = {F(extremum_F)}")
    else:
        print("Экстремум F(x) не найден")

    if extremum_P is not None:
        print(f"Экстремум P(x): {extremum_P}, P({extremum_P}) = {P(extremum_P)}")
    else:
        print("Экстремум P(x) не найден")

    # Найдем общие решения уравнения F(x) = P(x)
    coeffs = [a1[0] - a2[0], a1[1] - a2[1], a1[2] - a2[2]]

    if coeffs[0] == 0 and coeffs[1] == 0 and coeffs[2] == 0:
        # Бесконечно много решений
        return None

    if coeffs[0] == 0:
        # Линейное уравнение
        if coeffs[1] == 0:
            # Уравнение вида 0 = const
            return []
        else:
            # Решение уравнения
            solution = -coeffs[2] / coeffs[1]
            return [(solution, F(solution))]

    # Решаем квадратное уравнение
    roots = np.roots(coeffs)
    results = []

    for root in roots:
        if np.isreal(root):
            root = root.real
            results.append((root, F(root)))

    # Сортируем результаты по x
    results.sort(key=lambda x: x[0])

    return results


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = np.mean(x)

    # Центральный момент третьего порядка
    m3 = np.sum((x - mean) ** 3) / n

    # Стандартное отклонение
    std_dev = np.std(x)

    # Коэффициент асимметрии
    A = m3 / (std_dev ** 3) if std_dev != 0 else 0

    return round(A, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = np.mean(x)

    # Центральный момент четвертого порядка
    m4 = np.sum((x - mean) ** 4) / n

    # Стандартное отклонение
    std_dev = np.std(x)

    # Коэффициент эксцесса
    E = (m4 / (std_dev ** 4)) - 3 if std_dev != 0 else 0

    return round(E, 2)
