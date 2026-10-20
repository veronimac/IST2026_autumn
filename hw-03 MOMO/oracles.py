import numpy as np
import scipy
import scipy.sparse
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Parameters
    ----------
        A : np.array or scipy.sparse.csr_matrix of shape (m, n)
            Feature matrix (training sample).
        b : np.array of shape (m,)
            Labels vector with values in {-1, 1}.
        regcoef : float
            Regularization coefficient.
    """
    def __init__(self, A, b, regcoef):
        self.A = A
        self.b = b
        self.regcoef = regcoef
        self._m = A.shape[0]

    def func(self, x):
        # Считаем z_i = b_i * <a_i, x> сразу для всех объектов выборки.
        # A.dot(x) - это вектор из скалярных произведений <a_i, x> (работает и для обычной, и для разреженной матрицы).
        # b * (...) - поэлементное умножение на метки классов (-1 или 1).
        z = self.b * self.A.dot(x)
        # np.logaddexp(0, -z) = ln(e^0 + e^(-z)) = ln(1 + e^(-z)).
        # Так безопаснее, чем писать np.log(1 + np.exp(-z)) в лоб: там при большом -z получилось бы переполнение (inf).
        # np.mean делит сумму на m - это множитель 1/m из формулы.
        loss = np.mean(np.logaddexp(0, -z))
        # Слагаемое регуляризации: (lambda / 2) * ||x||^2, а x.dot(x) как раз равно ||x||^2.
        reg = 0.5 * self.regcoef * x.dot(x)
        # Значение функции = потери + регуляризация.
        return loss + reg

    def grad(self, x):
        # Опять считаем z = b * (A x), он понадобится внутри сигмоиды.
        z = self.b * self.A.dot(x)
        # expit(t) = 1 / (1 + e^(-t)) - это сигмоида, у неё в scipy нет проблем с переполнением.
        # По формуле нужен вектор b * sigma(-z) (поэлементное произведение).
        w = self.b * expit(-z)
        # A.T.dot(w) = A^T w. Делим на m и ставим минус, как в формуле: -(1/m) * A^T (b * sigma(-z)).
        grad_loss = -self.A.T.dot(w) / self._m
        # Производная регуляризации (lambda/2)||x||^2 равна lambda * x.
        return grad_loss + self.regcoef * x

    def hess(self, x):
        # Снова z = b * (A x).
        z = self.b * self.A.dot(x)
        # sigma(-z) - вектор сигмоид, считаем один раз и используем дважды.
        sig = expit(-z)
        # s = sigma(-z) * (1 - sigma(-z)) - это диагональ матрицы Diag(s) из формулы гессиана.
        s = sig * (1.0 - sig)
        # Нам нужна матрица A^T * Diag(s) * A. Diag(s) в явном виде не строим (это матрица m x m, слишком большая),
        # а просто умножаем каждую строку A на соответствующее s_i.
        if scipy.sparse.issparse(self.A):
            # Для разреженной матрицы умножаем строки методом multiply.
            # s[:, None] превращает вектор длины m в столбец m x 1, чтобы умножение шло именно по строкам.
            weighted_A = self.A.multiply(s[:, None])
            # A^T * (взвешенная A). Результат разреженный, поэтому превращаем его в обычный np.array через toarray().
            # (Тест требует именно np.ndarray, а не np.matrix или sparse.)
            hess_loss = self.A.T.dot(weighted_A)
            hess_loss = np.asarray(hess_loss.toarray())
        else:
            # Для обычной матрицы то же самое делается простым умножением с broadcasting.
            weighted_A = self.A * s[:, None]
            hess_loss = self.A.T.dot(weighted_A)
        # Делим на m (множитель 1/m).
        hess_loss = hess_loss / self._m
        # Добавляем lambda * I (единичная матрица размера n x n, n = число признаков = длина x).
        return hess_loss + self.regcoef * np.eye(x.size)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)       # на всякий случай делаем из x массив чисел с плавающей точкой
    n = x.size                           # размерность задачи
    f0 = func(x)                         # значение функции в самой точке x (одно и то же для всех координат)
    grad = np.zeros(n)                   # сюда будем записывать частные производные
    for i in range(n):                   # идём по координатам по очереди
        e_i = np.zeros(n)                # базисный вектор e_i: везде нули...
        e_i[i] = 1.0                     # ...и единица на i-м месте
        # (f(x + eps*e_i) - f(x)) / eps - формула из условия: приближение i-й частной производной
        grad[i] = (func(x + eps * e_i) - f0) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x = np.asarray(x, dtype=float)       # приводим x к массиву float
    n = x.size                           # размерность
    f0 = func(x)                         # f(x) - считаем один раз
    # Чтобы не пересчитывать одни и те же значения, заранее считаем f(x + eps*e_i) для всех i.
    f_i = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)                # базисный вектор e_i
        e_i[i] = 1.0
        f_i[i] = func(x + eps * e_i)     # f(x + eps*e_i)
    hess = np.zeros((n, n))              # матрица Гессе n x n, пока из нулей
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        for j in range(i, n):            # гессиан симметричный, поэтому считаем только j >= i
            e_j = np.zeros(n)
            e_j[j] = 1.0
            f_ij = func(x + eps * e_i + eps * e_j)   # f(x + eps*e_i + eps*e_j)
            # формула из условия: (f(x+e_i+e_j) - f(x+e_i) - f(x+e_j) + f(x)) / eps^2
            value = (f_ij - f_i[i] - f_i[j] + f0) / (eps ** 2)
            hess[i, j] = value
            hess[j, i] = value           # зеркально записываем в симметричный элемент
    return hess
