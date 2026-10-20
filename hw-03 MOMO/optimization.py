import numpy as np
from numpy.linalg import LinAlgError
import scipy
import scipy.linalg
import scipy.optimize
import scipy.sparse
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # ---- 1) Постоянный шаг: ничего считать не надо, просто возвращаем число c ----
        if self._method == 'Constant':
            return self.c

        # ---- 2) Сильные условия Вульфа: берём готовую функцию из scipy ----
        if self._method == 'Wolfe':
            # scipy.optimize.line_search ищет шаг по сильным условиям Вульфа.
            # Ей нужны: функция, её градиент, точка x_k, направление d_k и константы c1, c2.
            result = scipy.optimize.line_search(oracle.func, oracle.grad, x_k, d_k,
                                                c1=self.c1, c2=self.c2)
            # Результат - либо кортеж (тогда шаг лежит на месте 0), либо объект с полем .alpha
            # (в разных версиях scipy по-разному), поэтому аккуратно обрабатываем оба случая.
            if hasattr(result, 'alpha'):
                alpha = result.alpha
            else:
                alpha = result[0]
            # Если scipy нашла шаг - возвращаем его.
            if alpha is not None:
                return alpha
            # Если не нашла (alpha == None) - по условию задания переходим к дроблению шага (Армихо),
            # начиная с alpha_0.
            start_alpha = self.alpha_0
        # ---- 3) Метод Армихо (дробление шага) ----
        else:
            # Если нам передали previous_alpha, начинаем с него, иначе - с alpha_0.
            start_alpha = previous_alpha if previous_alpha is not None else self.alpha_0

        alpha = start_alpha                              # текущее значение шага
        phi_0 = oracle.func_directional(x_k, d_k, 0.0)    # phi(0) = f(x_k)
        dphi_0 = oracle.grad_directional(x_k, d_k, 0.0)   # phi'(0) = <grad f(x_k), d_k>
        # Условие Армихо: phi(alpha) <= phi(0) + c1 * alpha * phi'(0).
        # Пока оно НЕ выполняется (т.е. phi(alpha) больше правой части) - делим шаг пополам.
        while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * dphi_0:
            alpha = alpha / 2.0
            # Защита от бесконечного цикла: если шаг стал совсем крошечным, то подходящего шага не нашлось.
            if alpha < 1e-16:
                return None
        return alpha


def _is_bad(*values):
    """
    Вспомогательная функция: возвращает True, если среди переданных значений
    есть None, inf или nan (т.е. вычисления сломались).
    """
    for v in values:
        if v is None:                                   # None - значит что-то не посчиталось
            return True
        if not np.all(np.isfinite(v)):                  # inf или nan хотя бы в одном элементе
            return True
    return False


def _save_history(history, start_time, x_k, f_k, grad_norm, trace):
    """
    Вспомогательная функция: дописывает в history текущее состояние метода (если trace=True).
    """
    if not trace:                                       # если историю вести не просили - ничего не делаем
        return
    history['time'].append((datetime.now() - start_time).total_seconds())   # сколько секунд прошло с начала
    history['func'].append(f_k)                         # значение функции
    history['grad_norm'].append(grad_norm)              # норма градиента (обычная, не в квадрате)
    if x_k.size <= 2:                                   # траекторию храним только для 1D и 2D задач
        history['x'].append(np.copy(x_k))               # copy, чтобы потом x_k не изменил сохранённое


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict or None
        Dictionary with line search options, e.g. {'method': 'Armijo', 'c1': 1e-4}.
        See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    if line_search_options is None:
        line_search_options = {}
    line_search_tool = LineSearchTool(**line_search_options)
    x_k = np.copy(x_0)

    start_time = datetime.now()                         # запоминаем момент старта (для history['time'])
    x_k = x_k.astype(float)                             # работаем с числами float, чтобы не было проблем с int-массивами
    alpha = None                                        # шаг с прошлой итерации (пока его нет)
    grad_0_sq = None                                    # ||grad f(x_0)||^2 - понадобится в критерии остановки

    # Цикл идёт от 0 до max_iter включительно: так мы успеваем сделать max_iter шагов
    # и ещё раз проверить критерий остановки уже в последней точке.
    for k in range(max_iter + 1):
        f_k = oracle.func(x_k)                          # значение функции в текущей точке
        grad_k = oracle.grad(x_k)                       # градиент в текущей точке
        # Если что-то стало None / inf / nan - сообщаем об ошибке вычислений.
        if _is_bad(x_k, f_k, grad_k):
            if display:
                print('Iteration {}: computational error'.format(k))
            return x_k, 'computational_error', history
        grad_sq = float(np.dot(grad_k, grad_k))         # ||grad f(x_k)||^2 (квадрат нормы градиента)
        if k == 0:
            grad_0_sq = grad_sq                         # запоминаем начальное значение для критерия (1.1)
        grad_norm = np.sqrt(grad_sq)                    # обычная норма градиента - её кладём в history

        _save_history(history, start_time, x_k, f_k, grad_norm, trace)   # пишем историю
        if display:                                     # отладочный вывод, если попросили
            print('Iteration {}: f = {}, ||grad|| = {}'.format(k, f_k, grad_norm))

        # Критерий остановки (1.1): ||grad(x_k)||^2 <= tolerance * ||grad(x_0)||^2.
        if grad_sq <= tolerance * grad_0_sq:
            if display:
                print('Success: stopping criterion is satisfied')
            return x_k, 'success', history
        # Если мы уже сделали max_iter шагов, а критерий всё ещё не выполнен - сдаёмся.
        if k == max_iter:
            break

        d_k = -grad_k                                   # направление спуска - антиградиент
        # Адаптивный шаг: на первой итерации стартуем с alpha_0 (previous_alpha=None),
        # дальше начинаем дробление с удвоенного шага с прошлой итерации.
        previous_alpha = None if alpha is None else 2.0 * alpha
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        if _is_bad(alpha):                              # линейный поиск не справился
            if display:
                print('Iteration {}: line search failed'.format(k))
            return x_k, 'computational_error', history
        x_k = x_k + alpha * d_k                         # сам шаг метода: x_{k+1} = x_k + alpha * d_k

    if display:
        print('Maximum number of iterations exceeded')
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict or None
        Dictionary with line search options, e.g. {'method': 'Armijo', 'c1': 1e-4}.
        See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    if line_search_options is None:
        line_search_options = {}
    line_search_tool = LineSearchTool(**line_search_options)
    x_k = np.copy(x_0)

    start_time = datetime.now()                         # момент старта
    x_k = x_k.astype(float)                             # переводим в float
    grad_0_sq = None                                    # ||grad f(x_0)||^2 для критерия остановки

    for k in range(max_iter + 1):                       # как и в градиентном спуске: max_iter шагов + финальная проверка
        f_k = oracle.func(x_k)                          # значение функции
        grad_k = oracle.grad(x_k)                       # градиент
        # Проверяем, что всё посчиталось нормально (нет None, inf, nan).
        if _is_bad(x_k, f_k, grad_k):
            if display:
                print('Iteration {}: computational error'.format(k))
            return x_k, 'computational_error', history
        grad_sq = float(np.dot(grad_k, grad_k))         # квадрат нормы градиента
        if k == 0:
            grad_0_sq = grad_sq                         # запоминаем для критерия (1.1)
        grad_norm = np.sqrt(grad_sq)                    # обычная норма градиента

        _save_history(history, start_time, x_k, f_k, grad_norm, trace)   # пишем историю
        if display:
            print('Iteration {}: f = {}, ||grad|| = {}'.format(k, f_k, grad_norm))

        # Критерий остановки (1.1)
        if grad_sq <= tolerance * grad_0_sq:
            if display:
                print('Success: stopping criterion is satisfied')
            return x_k, 'success', history
        if k == max_iter:                               # шаги закончились, а точность не достигнута
            break

        hess_k = oracle.hess(x_k)                       # гессиан в текущей точке
        if scipy.sparse.issparse(hess_k):               # если вдруг гессиан разреженный - делаем его обычным
            hess_k = hess_k.toarray()
        hess_k = np.atleast_2d(hess_k)                  # в 1D-тестах гессиан может прийти как вектор - делаем матрицу 1x1
        if _is_bad(hess_k):                             # в гессиане оказались inf/nan
            if display:
                print('Iteration {}: computational error (hessian)'.format(k))
            return x_k, 'computational_error', history

        # Направление Ньютона d_k = -H^{-1} grad: решаем систему H d = -grad через разложение Холецкого.
        try:
            c_and_lower = scipy.linalg.cho_factor(hess_k)            # раскладываем H = L L^T
            d_k = scipy.linalg.cho_solve(c_and_lower, -grad_k)       # решаем систему с помощью разложения
        except (LinAlgError, ValueError):               # матрица не положительно определённая -> разложение не получилось
            if display:
                print('Iteration {}: newton direction error'.format(k))
            return x_k, 'newton_direction_error', history

        # Линейный поиск. Для метода Ньютона дробление всегда начинаем с шага 1 (previous_alpha=1.0).
        alpha = line_search_tool.line_search(oracle, x_k, d_k, 1.0)
        if _is_bad(alpha):
            if display:
                print('Iteration {}: line search failed'.format(k))
            return x_k, 'computational_error', history
        x_k = x_k + alpha * d_k                         # шаг: x_{k+1} = x_k + alpha * d_k

    if display:
        print('Maximum number of iterations exceeded')
    return x_k, 'iterations_exceeded', history
