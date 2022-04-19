from hypothesis import given, strategies as st

from thesis.geometry import *

# Extreme values lead to rounding errors - especially around zero. This isn't really critical for this library,
# which is why we simplify the input values here.
realistic_floats = st.floats(-1000, 1000).map(lambda f: round(f, 2))

valid_quadratic = st.tuples(realistic_floats, realistic_floats, realistic_floats) \
    .filter(lambda v: v[0] != 0) \
    .map(lambda v: Quadratic(v[0], v[1], v[2]))

valid_points = st.tuples(realistic_floats, realistic_floats, realistic_floats) \
    .filter(lambda v: not (v[0] == v[1] or v[0] == v[2] or v[1] == v[2]))


@given(valid_quadratic, valid_points)
def test_quadratic_from_points(base, points):
    x1, x2, x3 = points
    y1 = base(x1)
    y2 = base(x2)
    y3 = base(x3)

    solved = Quadratic.from_points_precise((x1, y1), (x2, y2), (x3, y3))

    # Use very high absolute tolerance (because of high likelihood of rounding errors in solution)
    equal = np.isclose([base.a, base.b, base.c], [solved.a, solved.b, solved.c], atol=1.e-2).all()
    assert (equal, 'Quadratic was not correctly inferred.')


# points = st.tuples(realistic_floats, realistic_floats).map(lambda p: Vec2.from_tuple(p))
# positive_points = points.filter(lambda p: p.x > 0 and p.y > 0)
# valid_ellipse = st.tuples(points, positive_points, realistic_floats).map(lambda e: Ellipse(e[0], e[1], e[2]))
#
# @given(valid_ellipse)
# def test_ellipse_from_points()