from functools import cache
from sympy import binomial, factorial, floor, Integer, Poly, Rational
from sympy.abc import x


@cache
def _eulerian_number(n, k):
    """Return the Eulerian number A(n,k)."""
    return sum(
        [(-1) ** i * binomial(n + 1, i) * (k + 1 - i) ** n for i in range(k + 1)]
    )


@cache
def _eulerian_poly(n, x):
    """Calculate Eulerian polynomial A_n(x), first 10 values are given explicitly."""
    if n == 0:
        return Poly(1, x)
    return Poly(sum([_eulerian_number(n, k - 1) * x**k for k in range(1, n + 1)]))


@cache
def _binomial_polynomial(d, k, x):
    """Calculate the binomial polynomial binomial(x + k, d), with x as the variable."""
    poly = Poly(Rational(1, factorial(d)), x, domain="QQ")
    for i in range(d):
        poly *= Poly(x + k - i, x)

    return poly


def _is_unimodal(iterable):
    """Return True if the iterable is unimodal, False otherwise."""
    i = 1
    while i < len(iterable) and iterable[i - 1] <= iterable[i]:
        i += 1
    while i < len(iterable) and iterable[i - 1] >= iterable[i]:
        i += 1
    return i == len(iterable)


def is_valid_h_star_vector(h: tuple[int]) -> bool:
    """Check if the given integer vector violates any known h*-vector inequalities.

    This method checks that the h*-vector satisfies a list of know inequalities.
    This checks that the vector satisfies certain necessary conditions, does not
    guarantee that the resulting vector is an h*-vector.

    The list of checked properties/inequalities are:
    * h_i is an integer for all i,
    * h_0 = 1,
    * h_i >= 0 for all i,
    * h_1 >= h_d,
    * h_2 + ... + h_i >= h_{d-i+1} + ... + h_{d-1} for all i in [2, floor(d/2)],
    * h_0 + ... + h_i <= h_{s-i} + ... + h_s for all i in [0, floor(s/2)],
    * if s = d, then h_1 <= h_i for all i up to d-1,
    * if s < d, then h_0 + h_1 <= h_{i-d+s} + ... + h_i for all i in [1, d-1].

    Returns:
        True if the vector satisfies all known inequalities, False otherwise.
    """
    if any(not isinstance(h_i, (int, Integer)) for h_i in h):
        return False

    if any(h_i < 0 for h_i in h):
        return False

    if h[0] != 1:
        return False

    # d and s are shorthand for dimension and degree respectively
    d = len(h) - 1
    for s in range(d, 0, -1):
        if h[s] != 0:
            break

    if h[1] < h[d]:
        return False

    for i in range(2, floor(d / 2) + 1):
        if not sum(h[k] for k in range(2, i + 1)) >= sum(
            h[k] for k in range(d - i + 1, d)
        ):
            # does not satisfy Stanley's dim inequality
            return False

    for i in range(0, floor(s / 2) + 1):
        if not sum(h[k] for k in range(0, i + 1)) <= sum(
            h[k] for k in range(s - i, s + 1)
        ):
            # does not satisfy Stanley's dim inequality
            return False

    if s == d:
        for i in range(1, d - 1):
            if not h[1] <= h[i]:
                return False
    else:
        for i in range(1, d):
            if not h[0] + h[1] <= sum(h[k] for k in range(i - d + s, i + 1)):
                return False  # pragma: no cover
                # could not find a counterexample for this case

    return True


def h_star_to_ehrhart_polynomial(dim: int, h_star_vector: tuple[int]) -> Poly:
    """Get the Ehrhart polynomial from the h*-polynomial."""
    return sum(
        [
            h_i * _binomial_polynomial(dim, dim - i, x)
            for i, h_i in enumerate(h_star_vector)
            if h_i != 0
        ]
    )


def ehrhart_to_h_star_polynomial(
    dim: int, ehrhart_coefficients: tuple[Rational]
) -> Poly:
    """Get the h*-polynomial from the Ehrhart polynomial."""
    return sum(
        [
            ehrhart_coefficients[i] * _eulerian_poly(i, x) * (1 - x) ** (dim - i)
            for i in range(dim + 1)
        ]
    ).simplify()


def h_star_vector_of_cartesian_product_from_h_star_vectors(
    h_star_vector_1: tuple[int], h_star_vector_2: tuple[int]
) -> tuple[int]:
    """
    Return the h-star vector of the cartesian product of two polytopes, given
    their h-star vectors.
    """
    dim_1 = len(h_star_vector_1) - 1
    dim_2 = len(h_star_vector_2) - 1
    dim = dim_1 + dim_2

    ehrhart_polynomial_1 = h_star_to_ehrhart_polynomial(dim_1, h_star_vector_1)
    ehrhart_polynomial_2 = h_star_to_ehrhart_polynomial(dim_2, h_star_vector_2)
    ehrhart_polynomial = ehrhart_polynomial_1 * ehrhart_polynomial_2
    x = ehrhart_polynomial.gens[0]
    ehrhart_coeffs = tuple(
        [ehrhart_polynomial.coeff_monomial(x**deg) for deg in range(dim + 1)]
    )
    h_star_poly = ehrhart_to_h_star_polynomial(dim, ehrhart_coeffs)
    h_star_vector = tuple(
        [h_star_poly.coeff_monomial(x**deg) for deg in range(dim + 1)]
    )

    return h_star_vector
