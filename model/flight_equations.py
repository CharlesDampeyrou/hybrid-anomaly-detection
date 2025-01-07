from torch import cos, sin


def x_eq_aero(
    m,
    jx,
    jy,
    jz,
    alpha,
    beta,
    xi,
    p,
    temp,
    v,
    cx,
    cy,
    cz,
    thrust,
    normalize_by_load_factor=True,
):
    """
    Function computing the acceleration equation error on the x axis.
    inputs :
        - m : mass of the aircraft
        - jx, jy, jz : charge factor on the plane axis
        - alpha : angle of incidence
        - beta : drift angle
        - xi : angle between the thrust force and the x_b axis
        - p : air pression
        - temp : air temperature
        - v : true air speed
        - cl : lift coefficient
        - cd : drag coefficient
        - cq : lateral force coefficient
        - thrust
    """
    if normalize_by_load_factor:
        return (
            thrust * cos(xi) / m / jx
            + 0.5
            * p
            / temp
            * v**2
            / m
            / jx
            * (
                -cx * cos(alpha) * cos(beta)
                - cy * cos(alpha) * sin(beta)
                + cz * sin(alpha)
            )
            - 1
        )
    else:
        return (
            thrust * cos(xi) / m
            + 0.5
            * p
            / temp
            * v**2
            / m
            * (
                -cx * cos(alpha) * cos(beta)
                - cy * cos(alpha) * sin(beta)
                + cz * sin(alpha)
            )
            - jx
        )


def y_eq_aero(
    m,
    jx,
    jy,
    jz,
    alpha,
    beta,
    xi,
    p,
    temp,
    v,
    cx,
    cy,
    cz,
    thrust,
    normalize_by_load_factor=True,
):
    if normalize_by_load_factor:
        return 0.5 * p / temp * v**2 / m / jy * (cx * sin(beta) - cy * cos(beta)) - 1
    else:
        return 0.5 * p / temp * v**2 / m * (cx * sin(beta) - cy * cos(beta)) - jy


def z_eq_aero(
    m,
    jx,
    jy,
    jz,
    alpha,
    beta,
    xi,
    p,
    temp,
    v,
    cx,
    cy,
    cz,
    thrust,
    normalize_by_load_factor=True,
):
    if normalize_by_load_factor:
        return (
            -thrust * sin(xi) / m / jz
            + 0.5
            * p
            / temp
            * v**2
            / m
            / jz
            * (
                -cx * sin(alpha) * cos(beta)
                - cy * sin(alpha) * sin(beta)
                - cz * cos(alpha)
            )
            + 1  # le facteur de charge est donné selon -z_b dans les données Dassault
        )
    else:
        return (
            -thrust * sin(xi) / m
            + 0.5
            * p
            / temp
            * v**2
            / m
            * (
                -cx * sin(alpha) * cos(beta)
                - cy * sin(alpha) * sin(beta)
                - cz * cos(alpha)
            )
            + jz  # le facteur de charge est donné selon -z_b dans les données Dassault
        )


def x_eq_aircraft(
    m,
    jx,
    jy,
    jz,
    alpha,
    beta,
    xi,
    p,
    temp,
    v,
    cx,
    cy,
    cz,
    thrust,
    normalize_by_load_factor=True,
):
    """
    Function computing the acceleration equation error on the x axis. The difference between
    this function and x_eq_aero is that this one uses a description of the aerodynamic force on
    the plane axis (instead of the aerodynamic referential).
    inputs :
        - m : mass of the aircraft
        - jx, jy, jz : charge factor on the plane axis
        - alpha : angle of incidence
        - beta : drift angle
        - xi : angle between the thrust force and the x_b axis
        - p : air pression
        - temp : air temperature
        - v : true air speed
        - cl : aero force coef on x_b
        - cd : aero force coef on y_b
        - cq : aero force coef on z_b
        - thrust
    """
    if normalize_by_load_factor:
        return thrust * cos(xi) / m / jx - 0.5 * p / temp * v**2 / m / jx * cx - 1
    else:
        return thrust * cos(xi) / m - 0.5 * p / temp * v**2 / m * cx - jx


def y_eq_aircraft(
    m,
    jx,
    jy,
    jz,
    alpha,
    beta,
    xi,
    p,
    temp,
    v,
    cx,
    cy,
    cz,
    thrust,
    normalize_by_load_factor=True,
):
    if normalize_by_load_factor:
        return -0.5 * p / temp * v**2 / m / jy * cy - 1
    else:
        return -0.5 * p / temp * v**2 / m * cy - jy


def z_eq_aircraft(
    m,
    jx,
    jy,
    jz,
    alpha,
    beta,
    xi,
    p,
    temp,
    v,
    cx,
    cy,
    cz,
    thrust,
    normalize_by_load_factor=True,
):
    if normalize_by_load_factor:
        return (
            -thrust * sin(xi) / m / jz - 0.5 * p / temp * v**2 / m / jz * cz + 1
        )  # +1 car le facteur de charge est mesuré selon -z_b
    else:
        return (
            -thrust * sin(xi) / m - 0.5 * p / temp * v**2 / m * cz + jz
        )  # +1 car le facteur de charge est mesuré selon -z_b


def stengel_x_eq(
    m,
    jx,
    jy,
    jz,
    alpha,
    beta,
    xi,
    p,
    temp,
    v,
    cx,
    cy,
    cz,
    thrust,
    air_molar_mass,
    gas_constant,
    wing_surface,
    normalize_by_load_factor=False,
    **kwargs,
):
    """
    Function computing the acceleration equation error on the x axis. The difference between
    this function and x_eq_aero is that this one uses the aerodynamic force in the same way
    as Stengel in his simulator.
    inputs :
        - m : mass of the aircraft
        - jx, jy, jz : charge factor on the plane axis
        - alpha : angle of incidence
        - beta : drift angle
        - xi : angle between the thrust force and the x_b axis
        - p : air pression
        - temp : air temperature
        - v : true air speed
        - cx : aero force coef on x_b
        - cy : aero force coef on y_b
        - cz : aero force coef on z_b
        - thrust
        - air_molar_mass
        - gas_constant
        - wing_surface
    """
    g = 9.81
    if normalize_by_load_factor:
        return (
            thrust * cos(xi) / m / jx
            + 0.5
            * p
            / temp
            * air_molar_mass
            / gas_constant
            * wing_surface
            * v**2
            / m
            / jx
            * (-cx * cos(alpha) + cz * sin(alpha))
            - g
        )
    else:
        return (
            thrust * cos(xi) / m
            + 0.5
            * p
            / temp
            * air_molar_mass
            / gas_constant
            * wing_surface
            * v**2
            / m
            * (-cx * cos(alpha) + cz * sin(alpha))
            - jx * g
        )


def stengel_y_eq(
    m,
    jx,
    jy,
    jz,
    alpha,
    beta,
    xi,
    p,
    temp,
    v,
    cx,
    cy,
    cz,
    thrust,
    air_molar_mass,
    gas_constant,
    wing_surface,
    normalize_by_load_factor=False,
    **kwargs,
):
    g = 9.81
    if normalize_by_load_factor:
        return (
            0.5
            * p
            / temp
            * air_molar_mass
            / gas_constant
            * wing_surface
            * v**2
            / m
            / jy
            * cy
            - g
        )
    else:
        return (
            0.5
            * p
            / temp
            * air_molar_mass
            / gas_constant
            * wing_surface
            * v**2
            / m
            * cy
            - jy * g
        )


def stengel_z_eq(
    m,
    jx,
    jy,
    jz,
    alpha,
    beta,
    xi,
    p,
    temp,
    v,
    cx,
    cy,
    cz,
    thrust,
    air_molar_mass,
    gas_constant,
    wing_surface,
    normalize_by_load_factor=False,
    **kwargs,
):
    g = 9.81
    if normalize_by_load_factor:
        return (
            -thrust * sin(xi) / m / jz
            + 0.5
            * p
            / temp
            * air_molar_mass
            / gas_constant
            * wing_surface
            * v**2
            / m
            / jz
            * (-cx * sin(alpha) - cz * cos(alpha))
            + g
        )  # +1 car le facteur de charge est mesuré selon -z_b
    else:
        return (
            -thrust * sin(xi) / m
            + 0.5
            * p
            / temp
            * air_molar_mass
            / gas_constant
            * wing_surface
            * v**2
            / m
            * (-cx * sin(alpha) - cz * cos(alpha))
            + jz * g
        )  # +1 car le facteur de charge est mesuré selon -z_b


def l_eq(
    ixx,
    iyy,
    izz,
    ixz,
    p,
    q,
    r,
    p_d,
    q_d,
    r_d,
    pression,
    temp,
    v,
    cl,
    cm,
    cn,
    air_molar_mass,
    gas_constant,
    wing_surface,
    b,
    c_bar,
    **kwargs,
):
    """
    Function computing the moment equation error on the x axis. The moment coefficients
    are taken to a multiplicative constant. For instance, the moment used on x is
    1/2 * P / T * cl * v**2 instead of 1 / 2 * rho * v**2 * S * cl * b.
    Under the perfect gas law, rho = M/R * P/T (M=molar mass of the air), so this change
    is equivalent to multiplying the moment coefficients by M/R * S * b.
    The equation is given in Stengel's Matlab code.
    Automatic differentiation can be used if the function is called with torch tensors.
    Parameters :
        - ixx, iyy, izz, ixz : inertia matrix coefficients
        - p, q, r : angular velocities
        - p_d, q_d, r_d : angular accelerations
        - l_moment, m_moment, n_moment : moments
        - air_molar_mass
        - gas_constant : gas constant from the ideal gas law
        - wing_surface : wing surface used to compute the aerodynamic forces
        - b : wingspan
        - c_bar : mean aerodynamic chord
    """
    l_moment = (
        0.5
        * pression
        / temp
        * air_molar_mass
        / gas_constant
        * wing_surface
        * b
        * cl
        * v**2
    )
    n_moment = (
        0.5
        * pression
        / temp
        * air_molar_mass
        / gas_constant
        * wing_surface
        * b
        * cn
        * v**2
    )
    return (
        izz * l_moment
        + ixz * n_moment
        - (ixz * (iyy - ixx - izz) * p + (ixz**2 + izz * (izz - iyy)) * r) * q
    ) / (ixx * izz - ixz**2) - p_d


def m_eq(
    ixx,
    iyy,
    izz,
    ixz,
    p,
    q,
    r,
    p_d,
    q_d,
    r_d,
    pression,
    temp,
    v,
    cl,
    cm,
    cn,
    air_molar_mass,
    gas_constant,
    wing_surface,
    b,
    c_bar,
    **kwargs,
):
    """
    Function computing the moment equation error on the y axis. The moment coefficients
    are taken to a multiplicative constant. For instance, the moment used on x is
    1/2 * P / T * cl * v**2 instead of 1 / 2 * rho * v**2 * S * cl * b.
    Under the perfect gas law, rho = M/R * P/T (M=molar mass of the air), so this change
    is equivalent to multiplying the moment coefficients by M/R * S * b.
    The equation is given in Stengel's Matlab code.
    Automatic differentiation can be used if the function is called with torch tensors.
    Parameters :
        - ixx, iyy, izz, ixz : inertia matrix coefficients
        - p, q, r : angular velocities
        - p_d, q_d, r_d : angular accelerations
        - l_moment, m_moment, n_moment : moments
        - air_molar_mass
        - gas_constant : gas constant from the ideal gas law
        - wing_surface : wing surface used to compute the aerodynamic forces
        - b : wingspan
        - c_bar : mean aerodynamic chord
    """
    m_moment = (
        0.5
        * pression
        / temp
        * air_molar_mass
        / gas_constant
        * wing_surface
        * c_bar
        * cm
        * v**2
    )
    return (m_moment - (ixx - izz) * p * r - ixz * (p**2 - r**2)) / iyy - q_d


def n_eq(
    ixx,
    iyy,
    izz,
    ixz,
    p,
    q,
    r,
    p_d,
    q_d,
    r_d,
    pression,
    temp,
    v,
    cl,
    cm,
    cn,
    air_molar_mass,
    gas_constant,
    wing_surface,
    b,
    c_bar,
    **kwargs,
):
    """
    Function computing the moment equation error on the z axis. The moment coefficients
    are taken to a multiplicative constant. For instance, the moment used on x is
    1/2 * P / T * cl * v**2 instead of 1 / 2 * rho * v**2 * S * cl * b.
    Under the perfect gas law, rho = M/R * P/T (M=molar mass of the air), so this change
    is equivalent to multiplying the moment coefficients by M/R * S * b.
    The equation is given in Stengel's Matlab code.
    Automatic differentiation can be used if the function is called with torch tensors.
    Parameters :
        - ixx, iyy, izz, ixz : inertia matrix coefficients
        - p, q, r : angular velocities
        - p_d, q_d, r_d : angular accelerations
        - l_moment, m_moment, n_moment : moments
        - air_molar_mass
        - gas_constant : gas constant from the ideal gas law
        - wing_surface : wing surface used to compute the aerodynamic forces
        - b : wingspan
        - c_bar : mean aerodynamic chord
    """
    l_moment = (
        0.5
        * pression
        / temp
        * air_molar_mass
        / gas_constant
        * wing_surface
        * b
        * cl
        * v**2
    )
    n_moment = (
        0.5
        * pression
        / temp
        * air_molar_mass
        / gas_constant
        * wing_surface
        * b
        * cn
        * v**2
    )
    return (
        ixz * l_moment
        + ixx * n_moment
        + (ixz * (iyy - ixx - izz) * r + (ixz**2 + ixx * (ixx - iyy)) * p) * q
    ) / (ixx * izz - ixz**2) - r_d
