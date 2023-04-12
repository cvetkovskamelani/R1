import math
import sympy as sp
import ipywidgets as widgets
import plotly.graph_objects as go


def point2(x=0, y=0):
    """Return a 2D point in Cartesian coordinates."""
    return sp.Matrix([x, y])


def hpoint2(x=0, y=0, w=1):
    """Return a 2D point in homogeneous coordinates."""
    return sp.Matrix([x, y, w])


def point3(x=0, y=0, z=0):
    """Return a 3D point in Cartesian coordinates."""
    return sp.Matrix([x, y, z])


def hpoint3(x=0, y=0, z=0, w=1):
    """Return a 3D point in homogeneous coordinates."""
    return sp.Matrix([x, y, z, w])


def unit_vector(vector, coordinates='Cartesian'):
    """
    Normalize the given vector.

    :param vector: is vector to be normalized.
    :type vector: Matrix
    ...
    :param coordinates: are coordinates in which the vector is described, 'Cartesian' or 'homogeneous'.
    :type coordinates: string
    ...
    :raise ValueError: if the value of coordinates is invalid.
    ...
    :return: the normalized vector.
    """
    if coordinates == 'Cartesian':
        return vector / math.sqrt(sum(vector.multiply_elementwise(vector)))
    elif coordinates == 'homogeneous':
        vector = vector[:-1, :]
        vector = vector / math.sqrt(sum(vector.multiply_elementwise(vector)))
        return sp.Matrix.vstack(vector, sp.Matrix([0]))
    raise ValueError(f'Invalid coordinates value. Expected "Cartesian" or "homogeneous" but "{coordinates}:was given.')


def rotation3(axis, angle):
    """
    Return a rotation matrix in 3D Cartesian coordinates that represents the rotation by `angle` around `axis`.

    :param axis: is the axis of direction. It should be one of ['x', 'y', 'z', 'n', 'o', 'a'].
    :type axis: string
    ...
    :param angle: is the angle of rotation.
    :type angle: sympy expression, sympy symbol or a number
    ...
    raise ValueError: if an invalid axis value is received.
    ...
    :return: the rotation matrix
    """
    if axis in ['x', 'n']:
        return sp.Matrix([
            [1, 0, 0],
            [0, sp.cos(angle), -sp.sin(angle)], 
            [0, sp.sin(angle), sp.cos(angle)]
        ])
    elif axis in ['y', 'o']:
        return sp.Matrix([
            [sp.cos(angle), 0, sp.sin(angle)],
            [0, 1, 0],
            [-sp.sin(angle), 0, sp.cos(angle)]
        ])
    elif axis in ['z', 'a']:
        return sp.Matrix([
            [sp.cos(angle), -sp.sin(angle), 0], 
            [sp.sin(angle), sp.cos(angle), 0], 
            [0, 0, 1]
        ])
    else:
        raise ValueError(f'Expected one of [x, y, z, n, o, a] but received {axis}.')


def hrotation3(axis, angle):
    """
    Return a transformation matrix in 3D homogeneous coordinates that represents the rotation by `angle` around `axis`.

    :param axis: is the axis of rotation. It should be one of ['x', 'y', 'z', 'n', 'o', 'a'].
    :type axis: string
    ...
    :param angle: is the angle of rotation.
    :type angle: sympy expression, sympy symbol or a number
    ...
    raise ValueError: if an invalid axis value is received.
    ...
    :return: the rotation matrix
    """
    if axis in ['x', 'n']:
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, sp.cos(angle), -sp.sin(angle), 0], 
            [0, sp.sin(angle), sp.cos(angle), 0],
            [0, 0, 0, 1]
        ])
    elif axis in ['y', 'o']:
        return sp.Matrix([
            [sp.cos(angle), 0, sp.sin(angle), 0],
            [0, 1, 0, 0],
            [-sp.sin(angle), 0, sp.cos(angle), 0],
            [0, 0, 0, 1]
        ])
    elif axis in ['z', 'a']:
        return sp.Matrix([
            [sp.cos(angle), -sp.sin(angle), 0, 0], 
            [sp.sin(angle), sp.cos(angle), 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError(f'Expected one of [x, y, z, n, o, a] but received {axis}.')

        
def rotation2(angle):
    """Return a rotation matrix in 2D Cartesian coordinates that represents the rotation by the given angle."""
    return sp.Matrix([
        [sp.cos(angle), -sp.sin(angle)], 
        [sp.sin(angle), sp.cos(angle)]
    ])


def hrotation2(angle):
    """Return a rotation matrix in 2D homogeneous coordinates that represents the rotation by the given angle."""
    return sp.Matrix([
        [sp.cos(angle), -sp.sin(angle), 0], 
        [sp.sin(angle), sp.cos(angle), 0], 
        [0, 0, 1]
    ])


def htranslation3(x=0, y=0, z=0):
    """Return transformation matrix in 3D homogeneous coordinates with embedded translation."""
    return sp.Matrix([[1, 0, 0, x],
                      [0, 1, 0, y],
                      [0, 0, 1, z],
                      [0, 0, 0, 1]])


def htranslation2(x=0, y=0):
    """Return transformation matrix in 2D homogeneous coordinates with embedded translation."""
    return sp.Matrix([[1, 0, x],
                      [0, 1, y],
                      [0, 0, 1]])


def is_right_hand_coordinate_system3(pose):
    """Checks whether the given pose follows the right-hand rule."""   
    M = pose[:3, :3]
    return sp.simplify(M*M.transpose()) == sp.eye(3) and sp.simplify(sp.det(M)) == 1


def hpose3(x=(1, 0, 0), y=(0, 1, 0), z=(0, 0, 1), t=(0, 0, 0)):
    """Return a pose in 3D homogeneous coordinates where the arguments are the new frame's axes and translation vector."""
    return sp.Matrix.vstack(sp.Matrix([tuple(x), tuple(y), tuple(z), tuple(t)]).T, sp.Matrix([0, 0, 0, 1]).T)


def euler_angles(pose, sequence):
    """
    Return the euler angles for the given rotation.

    :param pose: is a rotation or transformation matrix.
    :type pose: Matrix
    ...
    :param sequence: is the euler angles configuration.
    :type sequence: string 
    ...
    raise ValueError: if an invalid or unsupported euler angle sequence is received.
    ...
    :return: the euler angles.
    """
#     Ова се дел од Ојлеровите агли.
#     За останатите оди на https://www.geometrictools.com/Documentation/EulerAngles.pdf
    sequence = sequence.lower()
    if sequence == 'zyz':
        if pose[2, 2] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(pose[1, 0], pose[1, 1]), 0, 0
        elif pose[2, 2] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(pose[1, 0], pose[1, 1]), sp.pi, 0
        else:
            theta_1 = sp.atan2(pose[1, 2], pose[0, 2])
            theta_2 = sp.acos(pose[2, 2])
            theta_3 = sp.atan2(pose[2, 1], -pose[2, 0])
        return sp.Matrix([theta_1, theta_2, theta_3])
    elif sequence == 'zyx':
        if pose[2, 0] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(-pose[1, 2], pose[1, 1]), -sp.pi/2, 0
        elif pose[2, 0] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(-pose[1, 2], pose[1, 1]), sp.pi/2, 0
        else:
            theta_1 = sp.atan2(pose[1, 0], pose[0, 0])
            theta_2 = sp.asin(-pose[2, 0])
            theta_3 = sp.atan2(pose[2, 1], pose[2, 2])
        return sp.Matrix([theta_1, theta_2, theta_3])
    elif sequence == 'xyz':
        if pose[0, 2] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(pose[1, 0], pose[1, 1]), sp.pi/2, 0
        elif pose[0, 2] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(pose[1, 0], pose[1, 1]), -sp.pi/2, 0
        else:
            theta_1 = sp.atan2(-pose[1, 2], pose[2, 2])
            theta_2 = sp.asin(pose[0, 2])
            theta_3 = sp.atan2(-pose[0, 1], pose[0, 0])
        return sp.Matrix([theta_1, theta_2, theta_3])
    elif sequence == 'xzy':
        if pose[0, 1] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(-pose[2, 0], pose[2, 2]), -sp.pi/2, 0
        elif pose[0, 1] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(-pose[2, 0], pose[2, 2]), sp.pi/2, 0
        else:
            theta_1 = sp.atan2(pose[2, 1], pose[1, 1])
            theta_2 = sp.asin(-pose[0, 1])
            theta_3 = sp.atan2(pose[0, 2], pose[0, 0])
        return sp.Matrix([theta_1, theta_2, theta_3])
    elif sequence == 'xzx':
        if pose[0, 0] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(pose[2, 1], pose[2, 2]), 0, 0
        elif pose[0, 0] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(pose[2, 1], pose[2, 2]), sp.pi, 0
        else:
            theta_1 = sp.atan2(pose[2, 0], pose[1, 0])
            theta_2 = sp.acos(pose[0, 0])
            theta_3 = sp.atan2(pose[0, 2], -pose[0, 1])
        return sp.Matrix([theta_1, theta_2, theta_3])
    elif sequence == 'xyx':
        if pose[0, 0] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(-pose[1, 2], pose[1, 1]), 0, 0
        elif pose[0, 0] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(-pose[1, 2], pose[1, 1]), sp.pi, 0
        else:
            theta_1 = sp.atan2(pose[1, 0], -pose[2, 0])
            theta_2 = sp.acos(pose[0, 0])
            theta_3 = sp.atan2(pose[0, 1], pose[0, 2])
        return sp.Matrix([theta_1, theta_2, theta_3])
    elif sequence == 'yxz':
        if pose[1, 2] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(-pose[0, 1], pose[0, 0]), -sp.pi/2, 0
        elif pose[1, 2] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(-pose[0, 1], pose[0, 0]), sp.pi/2, 0
        else:
            theta_1 = sp.atan2(pose[0, 2], pose[2, 2])
            theta_2 = sp.asin(-pose[1, 2])
            theta_3 = sp.atan2(pose[1, 0], pose[1, 1])
        return sp.Matrix([theta_1, theta_2, theta_3])
    elif sequence == 'yzx':
        if pose[1, 0] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(pose[2, 1], pose[2, 2]), sp.pi/2, 0
        elif pose[1, 0] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(pose[2, 1], pose[2, 2]), -sp.pi/2, 0
        else:
            theta_1 = sp.atan2(-pose[2, 0], pose[0, 0])
            theta_2 = sp.asin(pose[1, 0])
            theta_3 = sp.atan2(-pose[1, 2], pose[1, 1])
        return sp.Matrix([theta_1, theta_2, theta_3])
    elif sequence == 'yxy':
        if pose[1, 1] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(pose[0, 2], pose[0, 0]), 0, 0
        elif pose[1, 1] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(pose[0, 2], pose[0, 0]), sp.pi, 0
        else:
            theta_1 = sp.atan2(pose[0, 1], pose[2, 1])
            theta_2 = sp.acos(pose[1, 1])
            theta_3 = sp.atan2(pose[1, 0], -pose[1, 2])
        return sp.Matrix([theta_1, theta_2, theta_3])
    elif sequence == 'yzy':
        if pose[1, 1] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(-pose[2, 0], pose[2, 2]), 0, 0
        elif pose[1, 1] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(-pose[2, 0], pose[2, 2]), sp.pi, 0
        else:
            theta_1 = sp.atan2(pose[2, 1], -pose[0, 1])
            theta_2 = sp.acos(pose[1, 1])
            theta_3 = sp.atan2(pose[1, 2], pose[1, 0])
        return sp.Matrix([theta_1, theta_2, theta_3])
    elif sequence == 'zxy':
        if pose[2, 1] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(pose[0, 2], pose[0, 0]), sp.pi/2, 0
        elif pose[2, 1] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(pose[0, 2], pose[0, 0]), -sp.pi/2, 0
        else:
            theta_1 = sp.atan2(-pose[0, 1], pose[1, 1])
            theta_2 = sp.asin(pose[2, 1])
            theta_3 = sp.atan2(-pose[2, 0], pose[2, 2])
        return sp.Matrix([theta_1, theta_2, theta_3])
    elif sequence == 'zxz':
        if pose[2, 2] == 1:
            theta_1, theta_2, theta_3 = sp.atan2(-pose[0, 1], pose[0, 0]), 0, 0
        elif pose[2, 2] == -1:
            theta_1, theta_2, theta_3 = -sp.atan2(-pose[0, 1], pose[0, 0]), sp.pi, 0
        else:
            theta_1 = sp.atan2(pose[0, 2], -pose[1, 2])
            theta_2 = sp.acos(pose[2, 2])
            theta_3 = sp.atan2(pose[2, 0], pose[2, 1])
        return sp.Matrix([theta_1, theta_2, theta_3])
    raise ValueError(f'Invalid or unsupported euler angle sequence {sequence}.')


def rotation_matrix_from_euler_angles(euler_angles, sequence):
    """
    Return a transformation matrix for the given euler angles.

    :param euler_angles: are the euler angles.
    :type euler_angles: Matrix
    ...
    :param sequence: is the euler angles configuration.
    :type sequence: string 
    ...
    raise ValueError: if an invalid or unsupported euler angle sequence is received.
    ...
    :return: a transformation matrix for the given euler angles.
    """
    sequence = sequence.lower()
    if not all([char in 'xyz' for char in sequence]) or len(sequence) != 3:
        raise ValueError(f'Invalid or unsupported euler angle sequence {sequence}.')
    T1 = hrotation3(sequence[0], euler_angles[0])
    T2 = hrotation3(sequence[1], euler_angles[1])
    T3 = hrotation3(sequence[2], euler_angles[2])
    return T1 * T2 * T3


def lagrangian(L, jvars):
    """ 
    Implementation of Lagrange's equations. 

    :param L: is the Lagrangian.
    :type L: Matrix
    ...
    :param jvars: is a list of all variables used in the Lagrangian.
    :type jvars: list or Matrix
    ...
    :return: the generalized forces equations.
    """
    t = sp.symbols('t')
    jvars = sp.Matrix(jvars)
    return sp.simplify(L.diff(jvars.diff(t)).diff(t) - L.diff(jvars))


def dynamic_model_with_4_matrices(L, jvars):
    """ 
    Create the dynamic model through implementation of Lagrange's equations as by the book.

    :param L: is the Lagrangian.
    :type L: Matrix
    ...
    :param jvars: is a list of all variables used in the Lagrangian.
    :type jvars: list or Matrix
    ...
    :return: the 4 matrices that create the dynamic model.
    """
    import itertools
    from sympy.core.add import Add

    t = sp.symbols('t')
    n = len(jvars)
    M, Cf, Ck, G = sp.zeros(n), sp.zeros(n), sp.zeros(n, n*(n-1)//2), sp.zeros(n, 1)
    jvars = sp.Matrix(jvars)
    v_jvars, a_jvars = jvars.diff(t), jvars.diff(t, 2)
    v_jvars_squared = sp.matrix_multiply_elementwise(v_jvars, v_jvars)
    Ck_vars = [var[0] * var[1] for var in itertools.combinations(v_jvars, 2)]
    for row, eq in enumerate(lagrangian(L, jvars).expand()):
        args = eq.args if eq.func is Add else [eq]
        M[row, :] = [[sum([arg for arg in args if arg.has(var)]) for var in a_jvars]]
        Cf[row, :] = [[sum([arg for arg in args if arg.has(var)]) for var in v_jvars_squared]]
        if n > 1:
            Ck[row, :] = [[sum([arg for arg in args if arg.has(var)]) for var in Ck_vars]]
        G[row, :] = [sum([arg for arg in args if not arg.has(*a_jvars) and not arg.has(*v_jvars)])]
    return sp.simplify(M), sp.simplify(Cf), sp.simplify(Ck), sp.simplify(G)


def dynamic_model_with_3_matrices(L, jvars):
    """ 
    Create the dynamic model through implementation of Lagrange's equations as by the book.

    :param L: is the Lagrangian.
    :type L: Matrix
    ...
    :param jvars: is a list of all variables used in the Lagrangian.
    :type jvars: list or Matrix
    ...
    :return: the 3 matrices that create the dynamic model.
    """
    import itertools
    from sympy.core.add import Add

    t = sp.symbols('t')
    n = len(jvars)
    M, C, G = sp.zeros(n), sp.zeros(n, n*(n+1)//2), sp.zeros(n, 1)
    jvars = sp.Matrix(jvars)
    v_jvars, a_jvars = jvars.diff(t), jvars.diff(t, 2)
    C_vars = [var[0] * var[1] for var in itertools.combinations_with_replacement(v_jvars, 2)]
    for row, eq in enumerate(lagrangian(L, jvars).expand()):
        args = eq.args if eq.func is Add else [eq]
        M[row, :] = [[sum([arg for arg in args if arg.has(var)]) for var in a_jvars]]
        C[row, :] = [[sum([arg for arg in args if arg.has(var)]) for var in C_vars]]
        G[row, :] = [sum([arg for arg in args if not arg.has(*a_jvars) and not arg.has(*v_jvars)])]
    return sp.simplify(M), sp.simplify(C), sp.simplify(G)


def force_transformation_matrix(pose):
    """ Return the force transformation matrix for the given pose."""
    R, P = pose[:3, :3], pose[:3, 3]
    return sp.Matrix.vstack(
        sp.Matrix.hstack(R, sp.zeros(3)),
        sp.Matrix.hstack(delta3(*P)*R, R))


def dh_joint_to_joint(theta, d, a, alpha):
    """ Get the DH model elementary matrix consisted of theta, d, a, apha parameters. """
    return hrotation3('z', theta) * htranslation3(x=a, z=d) * hrotation3('x', alpha)


def frame_lines(pose, line_length=1):
    pose = pose.evalf()
    t = pose[:3, -1]
    line_n = sp.Matrix.hstack(t, t + line_length * pose[:3, 0])
    line_o = sp.Matrix.hstack(t, t + line_length * pose[:3, 1])
    line_a = sp.Matrix.hstack(t, t + line_length * pose[:3, 2])
    return line_n, line_o, line_a


def draw_frame(pose, labels='x-y-z', colors=('red', 'green', 'blue'), line_width=3, line_length=1):
    def go_scatter(x, y, z, name, color, width):
        return go.Scatter3d(
            x=x, y=y, z=z,
            marker=dict(size=2),
            line=dict(color=color, width=width),
            name=name)

    labels = labels.split('-')
    line_n, line_o, line_a = frame_lines(pose, line_length)
    scatter_n = go_scatter(*line_n.tolist(), labels[0], colors[0], line_width)
    scatter_o = go_scatter(*line_o.tolist(), labels[1], colors[1], line_width)
    scatter_a = go_scatter(*line_a.tolist(), labels[2], colors[2], line_width)
    return scatter_n, scatter_o, scatter_a


def plot(pose=hpose3(), point=hpoint3()):
    scatter_xyz = draw_frame(hpose3(), labels='x-y-z', colors=('red', 'green', 'blue'), line_width=3, line_length=1.5)
    scatter_noa = draw_frame(pose, labels='n-o-a', colors=('cyan', 'magenta', 'yellow'), line_width=6)
    point = point.tolist()
    scatter_point = go.Scatter3d(x=point[0], y=point[1], z=point[2], marker=dict(size=2), line=dict(color='black'), name='p')
    plot_data = scatter_xyz + scatter_noa + (scatter_point,)
    fig = go.FigureWidget(data=plot_data)
    fig.update_layout(width=800, height=600, autosize=False, scene=dict(aspectmode='data'))
    return fig


def delta3(dRx=0, dRy=0, dRz=0):
    """
    Return the differential operator for rotation matrix in 3D.

    :params dRx, dRy, dRz: are the differential rotations.
    :type dRx, dRy, dRz: Symbol or number
    ...
    :return: the differential operator for rotation matrix in 3D.
    """
    return sp.Matrix([
        [0, -dRz, dRy],
        [dRz, 0, -dRx],
        [-dRy, dRx, 0]
    ])


def hdelta3(dRx=0, dRy=0, dRz=0, dx=0, dy=0, dz=0):
    """
    Return the differential operator for homogeneous matrix in 3D.

    :params dRx, dRy, dRz: are the differential rotations.
    :type dRx, dRy, dRz: Symbol or number
    ...
    :params dx, dy, dz: are the differential translations.
    :type dx, dy, dz: Symbol or number 
    ...
    :return: the differential operator for homogeneous matrix in 3D.
    """
    return sp.Matrix([
        [0, -dRz, dRy, dx],
        [dRz, 0, -dRx, dy],
        [-dRy, dRx, 0, dz],
        [0, 0, 0, 0],
    ])


def extract_hdelta3(delta):
    """
    Return a vector consisted of differential rotations and translations.
    The vector elements are [δx, δy, δz, dx, dy, dz].

    :param delta: is the differential operator.
    :type delta: Matrix
    ...
    raise ValueError: if an invalid delta operator is received.
    ...
    :return: the vector of differential rotations and translations.
    """
    if delta[2, 1] != -delta[1, 2] or delta[2, 1] != -delta[1, 2] or delta[2, 1] != -delta[1, 2]:
        raise ValueError('delta is not a valid differential operator.')
    dRx, dRy, dRz = delta[2, 1], delta[0, 2], delta[1, 0]
    dx, dy, dz = delta[0, 3], delta[1, 3], delta[2, 3]
    return sp.Matrix([dRx, dRy, dRz, dx, dy, dz])


def polynomial(n, char='c'):
    """ Create an n-degree polynomial with custom symbols / coefficients. """
    t = sp.symbols('t')
    symbols = [sp.symbols(f'{char}{i}') for i in range(n+1)]
    p = sum([t**i * symbols[i] for i in range(n+1)])
    return p, symbols


def trajectory_polynomial_3(ti, tf, xi, xf, vi, vf):
    """
    Create a trajectory with 3th degree polynomial.

    :param ti: is the initial time.
    :type ti: number
    ...
    :param tf: is the final time.
    :type tf: number
    ...
    :param xi: is the initial position.
    :type xi: number
    ...
    :param xf: is the final position.
    :type xf: number
    ...
    :param vi: is the initial velocity.
    :type vi: number
    ...
    :param vf: is the final velocity.
    :type vf: number
    ...
    :return: the coefficients and position equation.
    """
    t = sp.symbols('t')
    theta, symbols = polynomial(3)
    v_theta = theta.diff(t)
    eq_xi = theta.subs(t, ti) - xi
    eq_xf = theta.subs(t, tf) - xf
    eq_vi = v_theta.subs(t, ti) - vi
    eq_vf = v_theta.subs(t, tf) - vf
    solution = sp.linsolve([eq_xi, eq_xf, eq_vi, eq_vf], symbols)
    for sub in zip(symbols, solution.args[0]):
        theta = theta.subs(*sub)
    return theta, solution


def trajectory_polynomial_5(ti, tf, xi, xf, vi, vf, ai, af):
    """
    Create a trajectory with 5th degree polynomial.

    :param ti: is the initial time.
    :type ti: number
    ...
    :param tf: is the final time.
    :type tf: number
    ...
    :param xi: is the initial position.
    :type xi: number
    ...
    :param xf: is the final position.
    :type xf: number
    ...
    :param vi: is the initial velocity.
    :type vi: number
    ...
    :param vf: is the final velocity.
    :type vf: number
    ...
    :param ai: is the initial acceleration.
    :type ai: number
    ...
    :param af: is the final acceleration.
    :type af: number
    ...
    :return: the coefficients and position equation.
    """
    t = sp.symbols('t')
    theta, symbols = polynomial(5)
    v_theta = theta.diff(t)
    a_theta = theta.diff(t, 2)
    eq_xi = theta.subs(t, ti) - xi
    eq_xf = theta.subs(t, tf) - xf
    eq_vi = v_theta.subs(t, ti) - vi
    eq_vf = v_theta.subs(t, tf) - vf
    eq_ai = a_theta.subs(t, ti) - ai
    eq_af = a_theta.subs(t, tf) - af
    solution = sp.linsolve([eq_xi, eq_xf, eq_vi, eq_vf, eq_ai, eq_af], symbols)
    for sub in zip(symbols, solution.args[0]):
        theta = theta.subs(*sub)
    return theta, solution


def trajectory_linear_parabolic_segments(ti, tf, xi, xf, w=None, tb=None):
    """
    Create a linear trajectory with parabolic end-segments.

    Note: Only one argument of w and tb is enough to get numerical solution.

    :param ti: is the initial time.
    :type ti: number
    ...
    :param tf: is the final time.
    :type tf: number
    ...
    :param xi: is the initial position.
    :type xi: number
    ...
    :param xf: is the final position.
    :type xf: number
    ...
    :param w: is the constant velocity in the linear segment.
    :type w: number
    ...
    :param tb: is the duration of each parabolic segment.
    :type tb: number
    ...
    :return: the coefficients w and tb and position equation of each segment.
    """
    if w and tb is None:
        tb = (xi - xf + w*tf) / w
    elif w is None and tb:
        w = (xi - xf) / (tb - tf)
    elif w is None and tb is None:
        w, tb = sp.symbols('t_b, w')
    t = sp.symbols('t')
    theta_s1 = xi + w / (2*tb) * t**2
    theta_s2 = theta_s1.subs(t, tb) - w*tb + w*t
    theta_s3 = xf - w / (2*tb) * (tf-t)**2
    return theta_s1, theta_s2, theta_s3, w, tb
    

def plot_trajectories(segments):
    """
    Plot the trajectories of the given list of segments. 
    A segment is defined as a tuple (equation, ti, tf).

    :param segments: is a list of segments.
    :type segments: list
    ...
    :return: the figure to be displayed.
    """
    import numpy as np
    import plotly.graph_objects as go

    t = sp.symbols('t')
    x_data, v_data, a_data, time_data = np.array([]), np.array([]), np.array([]), np.array([])
    for theta, ti, tf in segments:
        time = np.linspace(ti, tf, int(100*(tf-ti)))
        x = sp.lambdify(t, theta)(time)
        v = sp.lambdify(t, theta.diff(t))(time)
        a = sp.lambdify(t, theta.diff(t, 2))(time)
        time_data = np.hstack((time_data, time))
        x_data = np.hstack((x_data, x*np.ones_like(time) if isinstance(x, int) or isinstance(x, float) else x)) 
        v_data = np.hstack((v_data, v*np.ones_like(time) if isinstance(v, int) or isinstance(v, float) else v))
        a_data = np.hstack((a_data, a*np.ones_like(time) if isinstance(a, int) or isinstance(a, float) else a))

    line_position = go.Scatter(x=time_data, y=x_data, name='Позиција')
    line_velocity = go.Scatter(x=time_data, y=v_data, name='Брзина')
    line_acceleration = go.Scatter(x=time_data, y=a_data, name='Забрзување')
    return go.Figure([line_position, line_velocity, line_acceleration])


class AnimationPlayground:
    def play(self, checkpoint_pose=hpose3(), point=hpoint3()):
        self.checkpoint_pose = checkpoint_pose
        self.point = point

        self.toggle_action = widgets.ToggleButtons(options=['Ротирај', 'Придвижи се'])
        self.floattext = widgets.BoundedFloatText(
            value=0, min=-180, max=180, step=1, layout={'width': 'min-content'})
        self.dropdown_axis = widgets.Dropdown(
            options=['x', 'y', 'z', 'n', 'o', 'a'], description='по оската', layout={'width': 'min-content'})
        self.progress = widgets.FloatProgress(value=0, min=0, max=1, step=1)
        self.play = widgets.Play(value=0, min=0, max=1, step=1, show_repeat=False, interval=1)
        self.button_next = widgets.Button(description='Следно', disabled=True)

        self.floattext.observe(self.update_play_max, 'value')
        self.play.observe(self.update_fig, 'value')
        self.button_next.on_click(self.on_button_next_clicked)

        widgets.jslink((self.play, 'value'), (self.progress, 'value'))
        widgets.jslink((self.play, 'max'), (self.progress, 'max'))

        widget_box1 = widgets.HBox([self.toggle_action, self.floattext, self.dropdown_axis])
        widget_box2 = widgets.HBox([self.progress, self.play, self.button_next])

        self.fig = plot(self.checkpoint_pose, point)
        self.animation_box = widgets.VBox([widget_box1, widget_box2, self.fig])
        return self.animation_box

    def set_noa(self, pose):
        line_n, line_o, line_a = frame_lines(pose)
        with self.fig.batch_update():
            self.fig.data[3].x, self.fig.data[3].y, self.fig.data[3].z = line_n.tolist()
            self.fig.data[4].x, self.fig.data[4].y, self.fig.data[4].z = line_o.tolist()
            self.fig.data[5].x, self.fig.data[5].y, self.fig.data[5].z = line_a.tolist()

    def set_point_noa(self, point):
        point = self.current_pose * point
        point = point.evalf().tolist()
        with self.fig.batch_update():
            self.fig.data[6].x, self.fig.data[6].y, self.fig.data[6].z = point[0]*2, point[1]*2, point[2]*2

    def on_button_next_clicked(self, button):
        if self.play.value != self.play.max:
            return
        self.checkpoint_pose = self.current_pose
        self.play.value = 0
        self.floattext.value = 0

    def update_fig(self, change):
        if self.toggle_action.value == 'Ротирај':
            angle = sp.sign(self.floattext.value) * change.new / 180 * sp.pi
            if self.dropdown_axis.value in ['x', 'y', 'z']:
                self.current_pose = hrotation3(self.dropdown_axis.value, angle) * self.checkpoint_pose
            else:
                self.current_pose = self.checkpoint_pose * hrotation3(self.dropdown_axis.value, angle)
        elif self.toggle_action.value == 'Придвижи се':
            length = sp.sign(self.floattext.value) * change.new
            axis = self.dropdown_axis.value.replace('n', 'x').replace('o', 'y').replace('a', 'z')
            t = htranslation3(**{axis: length})
            if self.dropdown_axis.value in ['x', 'y', 'z']:
                self.current_pose = t * self.checkpoint_pose
            else:
                self.current_pose = self.checkpoint_pose * t
        line_n, line_o, line_a = frame_lines(self.current_pose)
        current_point = self.current_pose * self.point
        current_point = current_point.evalf().tolist()
        with self.fig.batch_update():
            self.fig.data[3].x, self.fig.data[3].y, self.fig.data[3].z = line_n.tolist()
            self.fig.data[4].x, self.fig.data[4].y, self.fig.data[4].z = line_o.tolist()
            self.fig.data[5].x, self.fig.data[5].y, self.fig.data[5].z = line_a.tolist()
            self.fig.data[6].x, self.fig.data[6].y, self.fig.data[6].z = current_point[0]*2, current_point[1]*2, current_point[2]*2

        self.toggle_action.disabled = True if self.play.value != 0 else False
        self.floattext.disabled = True if self.play.value != 0 else False
        self.dropdown_axis.disabled = True if self.play.value != 0 else False
        self.button_next.disabled = True if self.play.value != self.play.max else False

    def update_play_max(self, change):
        self.play.max = abs(self.floattext.value)


class SerialLinkRobot:
    """
    A class to easily create and interact with robotic arm.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset the robotic arm data. """
        self.links = []
        self.joint_variables = []
        self.subs_joints = []
        self.subs_additional = []
        self.stop_update_for_slider_joint = False
        self.trail_color_index = 1
        self.trail_color_list = []

    def add_revolute_joint(self, theta, d, a, alpha):
        """
        Add a revolute joint to the robotic arm according to the DH convention.

        :param theta: is the angle of rotation around z-axis.
        :type theta: Symbol
        ...
        :param d: is the displacement along z-axis.
        :type d: number or Symbol
        ...
        :param a: the displacement along x-axis.
        :type a: number or Symbol
        ...
        :param alpha: the angle of rotation around x-axis.
        :type alpha: number or Symbol
        """
        self.links.append(('revolute', theta, d, a, alpha))
        self.joint_variables.append(theta)
        self.subs_joints.append((theta, 0))

    def add_prismatic_joint(self, theta, d, a, alpha):
        """
        Add a prismatic joint to the robotic arm according to the DH convention.

        :param theta: is the angle of rotation around z-axis.
        :type theta: number or Symbol
        ...
        :param d: is the displacement along z-axis.
        :type d: Symbol
        ...
        :param a: the displacement along x-axis.
        :type a: number or Symbol
        ...
        :param alpha: the angle of rotation around x-axis.
        :type alpha: number or Symbol
        """
        self.links.append(('prismatic', theta, d, a, alpha))
        self.joint_variables.append(d)
        self.subs_joints.append((d, 1))

    def add_subs(self, subs):
        """
        Add the symbol values for plotting purposes.

        :param subs: is a list of tuples, each consisted of a symbol and its value.
        :type subs: [(symbol1, value1), (symbol2, value2), ... (symbol3, value3)]
        """
        self.subs_additional = subs

    def get_dh_joint_to_joint(self, start_joint, end_joint):
        """
        Get the DH model subsection transformation matrix for the joint id range(start_joint, end_joint).

        :param start_joint: is the starting joint id of the desired dh model susbsection.
        :type start_joint: integer
        ...
        :param end_joint: is the final joint id of the desired dh model susbsection.
        :type end_joint: integer
        ...
        :return: DH model subsection transformation matrix for joint id range(start_joint, end_joint).
        """
        pose = hpose3()
        for link in self.links[start_joint:end_joint]:
            joint_type, theta, d, a, alpha = link
            pose = pose * dh_joint_to_joint(theta, d, a, alpha)
        pose.simplify()
        return pose

    def get_dh_matrix(self):
        """ Get the DH model transformation matrix for the whole robotic arm. """
        return self.get_dh_joint_to_joint(start_joint=0, end_joint=len(self.links))

    def get_dh_table(self):
        """ Return the DH table intended for visual purposes only. """
        return sp.Matrix(self.links)[:, 1:]

    def linear_jacobian(self):
        """ Return the linear jacobian for this robotic arm. """
        linear_jacobian = self.get_dh_matrix()[:3, 3].jacobian(self.joint_variables)
        linear_jacobian.simplify()
        return linear_jacobian

    def angular_jacobian(self):
        """ Return the angular jacobian for this robotic arm. """
        pose = hpose3()
        angular_jacobian = sp.Matrix([[], [], []])
        for link in self.links:
            joint_type, theta, d, a, alpha = link
            z_i_m1 = sp.Matrix([0, 0, 0]) if joint_type == 'prismatic' else pose[:3, 2]
            angular_jacobian = sp.Matrix.hstack(angular_jacobian, z_i_m1)
            pose = pose * dh_joint_to_joint(theta, d, a, alpha)

        angular_jacobian.simplify()
        return angular_jacobian

    def jacobian(self):
        """ Return the jacobian for this robotic arm. """
        return sp.Matrix.vstack(self.linear_jacobian(), self.angular_jacobian())

    def update_toggle_joint(self, change):
        self.stop_update_for_slider_joint = True
        joint_id = int(change.new.split()[-1]) - 1
        joint_type = self.links[joint_id][0]
        minn, maxx, step = (0, 5, 0.1) if joint_type == 'prismatic' else (-180, 180, 1)
        self.slider_joint.min, self.slider_joint.max, self.slider_joint.step = minn, maxx, step
        joint_value = self.subs_joints[joint_id][1]
        self.slider_joint.value = joint_value if joint_type == 'prismatic' else int(180 * joint_value / sp.pi.evalf())
        self.slider_joint.description = 'Призматичен' if joint_type == 'prismatic' else 'Ротационен'
        self.stop_update_for_slider_joint = False

    def update_slider_joint(self, change):
        if self.stop_update_for_slider_joint:
            return
        joint_id = int(self.toggle_joint.value.split()[-1]) - 1
        joint_type = self.links[joint_id][0]
        joint_value = change.new if joint_type == 'prismatic' else change.new * sp.pi / 180
        self.subs_joints[joint_id] = self.subs_joints[joint_id][0], joint_value
        pose = hpose3()
        subs = self.subs_joints + self.subs_additional
        with self.fig.batch_update():
            for index, link in enumerate(self.links):
                index *= 5
                joint_type, theta, d, a, alpha = link
                next_pose = pose * dh_joint_to_joint(theta, d, a, alpha)
                pose_for_rectangluar_robot_shape = pose * htranslation3(z=d)
                pose_numeric = pose.subs(subs).evalf()
                next_pose_numeric = next_pose.subs(subs).evalf()
                pose_for_rectangluar_robot_shape_numeric = pose_for_rectangluar_robot_shape.subs(subs).evalf()
                pose_second_cone_numeric = pose_for_rectangluar_robot_shape_numeric if joint_type =='prismatic' else pose_numeric

                line_n, line_o, line_a = frame_lines(pose_numeric, line_length=0.5)
                self.fig.data[index+0].x, self.fig.data[index+0].y, self.fig.data[index+0].z = line_n.tolist()
                self.fig.data[index+1].x, self.fig.data[index+1].y, self.fig.data[index+1].z = line_a.tolist()

                cone_xyz = [pose_numeric[0, 3]], [pose_numeric[1, 3]], [pose_numeric[2, 3]]
                self.fig.data[index+2].x, self.fig.data[index+2].y, self.fig.data[index+2].z = cone_xyz
                cone_uvw = [pose_numeric[0, 2]], [pose_numeric[1, 2]], [pose_numeric[2, 2]]
                self.fig.data[index+2].u, self.fig.data[index+2].v, self.fig.data[index+2].w = cone_uvw

                cone_xyz = [pose_second_cone_numeric[0, 3]], [pose_second_cone_numeric[1, 3]], [pose_second_cone_numeric[2, 3]]
                self.fig.data[index+3].x, self.fig.data[index+3].y, self.fig.data[index+3].z = cone_xyz
                cone_uvw = [-pose_numeric[0, 2]], [-pose_numeric[1, 2]], [-pose_numeric[2, 2]]
                self.fig.data[index+3].u, self.fig.data[index+3].v, self.fig.data[index+3].w = cone_uvw

                self.fig.data[index+4].x = [pose_numeric[0, 3], pose_for_rectangluar_robot_shape_numeric[0, 3], next_pose_numeric[0, 3]]
                self.fig.data[index+4].y = [pose_numeric[1, 3], pose_for_rectangluar_robot_shape_numeric[1, 3], next_pose_numeric[1, 3]]
                self.fig.data[index+4].z = [pose_numeric[2, 3], pose_for_rectangluar_robot_shape_numeric[2, 3], next_pose_numeric[2, 3]]
                pose = next_pose

            line_n, line_o, line_a = frame_lines(pose.subs(subs).evalf(), line_length=0.5)
            self.fig.data[-3].x, self.fig.data[-3].y, self.fig.data[-3].z = line_n.tolist()
            self.fig.data[-2].x, self.fig.data[-2].y, self.fig.data[-2].z = line_a.tolist()

            if self.toggle_trail.value:
                self.fig.data[-1].x = self.fig.data[-1].x + (line_n.tolist()[0][0],)
                self.fig.data[-1].y = self.fig.data[-1].y + (line_n.tolist()[1][0],)
                self.fig.data[-1].z = self.fig.data[-1].z + (line_n.tolist()[2][0],)
                self.trail_color_list += [self.trail_color_index]
                self.fig.data[-1].marker['color'] = self.trail_color_list

    def update_button_trail_color(self, button):
        self.trail_color_index += 1

    def update_toggle_trail(self, change):
        change.owner.description = 'Вклучи цртање' if change.new == False else 'Исклучи цртање'

    def update_button_remove_trail(self, button):
        self.trail_color_index = 1
        self.trail_color_list = []
        with self.fig.batch_update():
            self.fig.data[-1].x,  self.fig.data[-1].y,  self.fig.data[-1].z = [], [], []

    def interact(self, figure_width=800, figure_height=600):
        """ Interact with the constructed robot arm. """
        if not self.links:
            return 'Роботската рака нема зглобови, па нема што да се црта.'
        self.toggle_trail = widgets.ToggleButton(value=True, description='Исклучи цртање')
        self.toggle_trail.observe(self.update_toggle_trail, 'value')
        self.button_trail_color = widgets.Button(description='Промени боја')
        self.button_trail_color.on_click(self.update_button_trail_color)
        self.button_remove_trail = widgets.Button(description='Избриши патека')
        self.button_remove_trail.on_click(self.update_button_remove_trail)
        self.toggle_joint = widgets.ToggleButtons(options=[f'Зглоб {x+1}' for x in range(len(self.links))])
        self.toggle_joint.observe(self.update_toggle_joint, 'value')
        joint_value = self.subs_joints[0][1]
        value, minn, maxx, step = (joint_value, 0, 5, 0.1) if self.links[0][0] == 'prismatic' else (joint_value, -180, 180, 1)
        decription = 'Призматичен' if self.links[0][0] == 'prismatic' else 'Ротационен'
        self.slider_joint = widgets.FloatSlider(
            value=value, min=minn, max=maxx, step=step, continuous_update=True, description=decription)
        self.slider_joint.observe(self.update_slider_joint, 'value')

        fig = self.plot(figure_width, figure_height)
        trail_buttons = widgets.HBox([self.toggle_trail, self.button_trail_color, self.button_remove_trail])
        widget_box = widgets.VBox([trail_buttons, self.toggle_joint, self.slider_joint, fig])
        return widget_box

    def plot(self, figure_width=800, figure_height=600):
        scatter_data = self.get_plot_data()
        self.fig = go.FigureWidget(data=scatter_data)
        self.fig.update_layout(width=figure_width, height=figure_height, showlegend=False, scene=dict(aspectmode='data'))
        return self.fig

    def get_plot_data(self):
        pose = hpose3()
        scatter_data = []
        self.joints_values = []
        subs = self.subs_joints + self.subs_additional
        for index, link in enumerate(self.links):
            joint_type, theta, d, a, alpha = link
            next_pose = pose * dh_joint_to_joint(theta, d, a, alpha)
            pose_for_rectangluar_robot_shape = pose * htranslation3(z=d)
            pose_numeric = pose.subs(subs).evalf()
            next_pose_numeric = next_pose.subs(subs).evalf()
            pose_for_rectangluar_robot_shape_numeric = pose_for_rectangluar_robot_shape.subs(subs).evalf()
            pose_second_cone_numeric = pose_for_rectangluar_robot_shape_numeric if joint_type =='prismatic' else pose_numeric
            colorscales = ['YlGnBu', 'agsunset', 'blues', 'bluered', 'amp']
            colorscale = colorscales[index % len(colorscales)]

            scatter_pose = draw_frame(pose_numeric, labels=f'x{index}-y{index}-z{index}', line_width=5, line_length=0.5)
            joint_1 = go.Cone(
                x=[pose_numeric[0, 3]], y=[pose_numeric[1, 3]], z=[pose_numeric[2, 3]],
                u=[pose_numeric[0, 2]], v=[pose_numeric[1, 2]], w=[pose_numeric[2, 2]],
                anchor='center', name=f'J{index}', showscale=False, colorscale=colorscale, sizemode='absolute', sizeref=0.5)
            joint_2 = go.Cone(
                x=[pose_second_cone_numeric[0, 3]], y=[pose_second_cone_numeric[1, 3]], z=[pose_second_cone_numeric[2, 3]],
                u=[-pose_numeric[0, 2]], v=[-pose_numeric[1, 2]], w=[-pose_numeric[2, 2]],
                anchor='center', name='', showscale=False, colorscale=colorscale, sizemode='absolute', sizeref=0.5)
            line_link = go.Scatter3d(
                x=[pose_numeric[0, 3], pose_for_rectangluar_robot_shape_numeric[0, 3], next_pose_numeric[0, 3]], 
                y=[pose_numeric[1, 3], pose_for_rectangluar_robot_shape_numeric[1, 3], next_pose_numeric[1, 3]],
                z=[pose_numeric[2, 3], pose_for_rectangluar_robot_shape_numeric[2, 3], next_pose_numeric[2, 3]],
                mode='lines', name='', line=dict(color='black', width=1))
            scatter_data += scatter_pose[:1] + scatter_pose[2:] + (joint_1, joint_2, line_link)
            pose = next_pose

        scatter_pose = draw_frame(
            pose.subs(subs).evalf(), labels=f'x{index+1}-y{index+1}-z{index+1}', colors=('cyan', 'magenta', 'yellow'), 
            line_width=5, line_length=0.5)
        scatter_data += scatter_pose[:1] + scatter_pose[2:]
        x, y, z = scatter_pose[0].x[0], scatter_pose[0].y[0], scatter_pose[0].z[0]
        scatter_data += (go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', name='', marker_size=2, marker_cmin=0),)
        return scatter_data
