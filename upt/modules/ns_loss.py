import torch.nn as nn
import torch

# define darcy loss
def ns_loss(u, x_coor, y_coor, flag_pde):

    # define loss
    mse = nn.MSELoss()

    # compute pde residual
    u_x = torch.autograd.grad(outputs=u, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_xx = torch.autograd.grad(outputs=u_x, inputs=x_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_y = torch.autograd.grad(outputs=u, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_yy = torch.autograd.grad(outputs=u_y, inputs=y_coor, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    pde_residual = u_xx + u_yy + 10
    pde_loss = mse(pde_residual*flag_pde, torch.zeros_like(pde_residual))

    return pde_loss


# class NavierStokes(PDE):
#     """
#     Compressible Navier Stokes equations
#     Reference:
#     https://turbmodels.larc.nasa.gov/implementrans.html

#     Parameters
#     ==========
#     nu : float, Sympy Symbol/Expr, str
#         The kinematic viscosity. If `nu` is a str then it is
#         converted to Sympy Function of form `nu(x,y,z,t)`.
#         If `nu` is a Sympy Symbol or Expression then this
#         is substituted into the equation. This allows for
#         variable viscosity.
#     rho : float, Sympy Symbol/Expr, str
#         The density of the fluid. If `rho` is a str then it is
#         converted to Sympy Function of form 'rho(x,y,z,t)'.
#         If 'rho' is a Sympy Symbol or Expression then this
#         is substituted into the equation to allow for
#         compressible Navier Stokes. Default is 1.
#     dim : int
#         Dimension of the Navier Stokes (2 or 3). Default is 3.
#     time : bool
#         If time-dependent equations or not. Default is True.
#     mixed_form: bool
#         If True, use the mixed formulation of the Navier-Stokes equations.

#     Examples
#     ========
#     >>> ns = NavierStokes(nu=0.01, rho=1, dim=2)
#     >>> ns.pprint()
#       continuity: u__x + v__y
#       momentum_x: u*u__x + v*u__y + p__x + u__t - 0.01*u__x__x - 0.01*u__y__y
#       momentum_y: u*v__x + v*v__y + p__y + v__t - 0.01*v__x__x - 0.01*v__y__y
#     >>> ns = NavierStokes(nu='nu', rho=1, dim=2, time=False)
#     >>> ns.pprint()
#       continuity: u__x + v__y
#       momentum_x: -nu*u__x__x - nu*u__y__y + u*u__x + v*u__y - 2*nu__x*u__x - nu__y*u__y - nu__y*v__x + p__x
#       momentum_y: -nu*v__x__x - nu*v__y__y + u*v__x + v*v__y - nu__x*u__y - nu__x*v__x - 2*nu__y*v__y + p__y
#     """

#     name = "NavierStokes"

#     def __init__(self, nu, rho=1, dim=3, time=True, mixed_form=False):
#         # set params
#         self.dim = dim
#         self.time = time
#         self.mixed_form = mixed_form

#         # coordinates
#         x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

#         # time
#         t = Symbol("t")

#         # make input variables
#         input_variables = {"x": x, "y": y, "z": z, "t": t}
#         if self.dim == 2:
#             input_variables.pop("z")
#         if not self.time:
#             input_variables.pop("t")

#         # velocity componets
#         u = Function("u")(*input_variables)
#         v = Function("v")(*input_variables)
#         if self.dim == 3:
#             w = Function("w")(*input_variables)
#         else:
#             w = Number(0)

#         # pressure
#         p = Function("p")(*input_variables)

#         # kinematic viscosity
#         if isinstance(nu, str):
#             nu = Function(nu)(*input_variables)
#         elif isinstance(nu, (float, int)):
#             nu = Number(nu)

#         # density
#         if isinstance(rho, str):
#             rho = Function(rho)(*input_variables)
#         elif isinstance(rho, (float, int)):
#             rho = Number(rho)

#         # dynamic viscosity
#         mu = rho * nu

#         # set equations
#         self.equations = {}
#         self.equations["continuity"] = (
#             rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
#         )

#         if not self.mixed_form:
#             curl = Number(0) if rho.diff(x) == 0 else u.diff(x) + v.diff(y) + w.diff(z)
#             self.equations["momentum_x"] = (
#                 (rho * u).diff(t)
#                 + (
#                     u * ((rho * u).diff(x))
#                     + v * ((rho * u).diff(y))
#                     + w * ((rho * u).diff(z))
#                     + rho * u * (curl)
#                 )
#                 + p.diff(x)
#                 - (-2 / 3 * mu * (curl)).diff(x)
#                 - (mu * u.diff(x)).diff(x)
#                 - (mu * u.diff(y)).diff(y)
#                 - (mu * u.diff(z)).diff(z)
#                 - (mu * (curl).diff(x))
#                 - mu.diff(x) * u.diff(x)
#                 - mu.diff(y) * v.diff(x)
#                 - mu.diff(z) * w.diff(x)
#             )
#             self.equations["momentum_y"] = (
#                 (rho * v).diff(t)
#                 + (
#                     u * ((rho * v).diff(x))
#                     + v * ((rho * v).diff(y))
#                     + w * ((rho * v).diff(z))
#                     + rho * v * (curl)
#                 )
#                 + p.diff(y)
#                 - (-2 / 3 * mu * (curl)).diff(y)
#                 - (mu * v.diff(x)).diff(x)
#                 - (mu * v.diff(y)).diff(y)
#                 - (mu * v.diff(z)).diff(z)
#                 - (mu * (curl).diff(y))
#                 - mu.diff(x) * u.diff(y)
#                 - mu.diff(y) * v.diff(y)
#                 - mu.diff(z) * w.diff(y)
#             )
#             self.equations["momentum_z"] = (
#                 (rho * w).diff(t)
#                 + (
#                     u * ((rho * w).diff(x))
#                     + v * ((rho * w).diff(y))
#                     + w * ((rho * w).diff(z))
#                     + rho * w * (curl)
#                 )
#                 + p.diff(z)
#                 - (-2 / 3 * mu * (curl)).diff(z)
#                 - (mu * w.diff(x)).diff(x)
#                 - (mu * w.diff(y)).diff(y)
#                 - (mu * w.diff(z)).diff(z)
#                 - (mu * (curl).diff(z))
#                 - mu.diff(x) * u.diff(z)
#                 - mu.diff(y) * v.diff(z)
#                 - mu.diff(z) * w.diff(z)
#             )

#             if self.dim == 2:
#                 self.equations.pop("momentum_z")