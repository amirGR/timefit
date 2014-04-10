"""Used offline for computing Poly.adjust_for_scaling()
   The resulting expressions were hard coded into Poly.adjust_for_scaling()
"""
from sympy import Symbol, expand, collect

def transform(degree):
    x = Symbol('x')
    A = Symbol('A')
    B = Symbol('B')
    a0 = Symbol('a0')
    xprime = A*(x-B)
    expr = a0
    for k in xrange(1,degree+1):
        a = Symbol('a{}'.format(k))
        expr = expr + a*xprime**k
    d = collect(expand(expr),x,evaluate=False)
    for k in xrange(degree+1):
        print '{}:\t{}'.format(x**k,d[x**k])
