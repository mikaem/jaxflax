import matplotlib.pyplot as plt
from functools import partial
import jax.numpy as jnp
from jax import Array
from jax import vmap
from jax import random
from jax import grad
from jax.tree_util import tree_map
from flax import nnx
import optax
import numpy as np
import jax
import time
from hessoptimizer import hess

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64

print("JAX running on", jax.devices()[0].platform.upper())

Re = 100.0
rho = 1.0
nu = 2.0 / Re
N = 64
factor = 5.0
# batch = 1
# xj = jnp.linspace(-1, 1, N + 1)
# xyi = jnp.cos(
#    jax.random.uniform(jax.random.PRNGKey(2003), (batch, (N + 1) * (N + 1), 2))
#    * 2
#    * jnp.pi
# )
# xyb = np.cos(
#    jax.random.uniform(jax.random.PRNGKey(2005), (batch, 4 * (N + 1), 2)) * 2 * np.pi
# )
# xyb[:, : N + 1, 1] = -1
# xyb[:, N + 1 : 2 * (N + 1), 1] = 1
# xyb[:, 2 * (N + 1) : 3 * (N + 1), 0] = -1
# xyb[:, 3 * (N + 1) :, 0] = 1

batch = 1
xj = jnp.cos(jnp.arange(N + 1) * jnp.pi / N)[::-1]
xi, yi = jnp.meshgrid(xj, xj, sparse=False, indexing="ij")
xyi = jnp.column_stack((xi[1:-1, 1:-1].ravel(), yi[1:-1, 1:-1].ravel()))[None, :, :]
xyb = jnp.vstack(
    (
        jnp.column_stack((xj, jnp.full(N + 1, -1))),
        jnp.column_stack((xj, jnp.full(N + 1, 1))),
        jnp.column_stack((jnp.full(N - 1, -1), xj[1:-1])),
        jnp.column_stack((jnp.full(N - 1, 1), xj[1:-1])),
    )
)[None, :, :]

xy0 = jnp.array(xyb)
xyp = jnp.array([0.0, 0.0])

t_data = (xyi, xy0, xyp)
# t_data = (xyi[:, :, 0], xy0[:, :, 0])

u0 = np.zeros((batch, xy0.shape[1], 3))
uj = (1 - xy0[:, N + 1 : 2 * (N + 1), 0]) ** 2 * (
    1 + xy0[:, N + 1 : 2 * (N + 1), 0]
) ** 2
u0[:, N + 1 : 2 * (N + 1), 0] = uj
# u0[N + 2 : 2 * (N + 1) - 1, 0] = 1
y_data = (jnp.zeros((batch, xyi.shape[1], 3)), jnp.array(u0), jnp.zeros(1))


class MLP(nnx.Module):
    def __init__(
        self, in_size: int, hidden_size: int, out_size: int, *, rngs: nnx.Rngs
    ) -> None:
        init = nnx.initializers.xavier_uniform(dtype=dtype)
        # init = nnx.initializers.xavier_normal()
        self.linear1 = nnx.Linear(
            in_size,
            hidden_size,
            rngs=rngs,
            kernel_init=init,
            param_dtype=dtype,
            dtype=dtype,
        )
        self.linear2 = nnx.Linear(
            hidden_size,
            hidden_size,
            rngs=rngs,
            kernel_init=init,
            param_dtype=dtype,
            dtype=dtype,
        )
        self.linear3 = nnx.Linear(
            hidden_size,
            out_size,
            rngs=rngs,
            kernel_init=init,
            param_dtype=dtype,
            dtype=dtype,
        )

    def __call__(self, x: Array) -> Array:
        x = nnx.tanh(self.linear1(x))
        x = nnx.tanh(self.linear2(x))
        return self.linear3(x)


model = MLP(2, 32, 3, rngs=nnx.Rngs(2002))

optlbfgs = optax.lbfgs(
    memory_size=100,
    linesearch=optax.scale_by_zoom_linesearch(100, max_learning_rate=1.0),
)
optadam = optax.adam(optax.linear_schedule(1e-3, 1e-4, 10000))
# optsgd = optax.inject_hyperparams(optax.sgd)(learning_rate=jnp.array([0.005]))
opthess = hess(
    use_lstsq=False,
    cg_max_iter=60,
    linesearch=optax.scale_by_zoom_linesearch(100, max_learning_rate=1.0),
)

opt_adam = nnx.Optimizer(model, optadam)
opt_lbfgs = nnx.Optimizer(model, optlbfgs)
opt_hess = nnx.Optimizer(model, opthess)


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x: [Array, Array, Array],
    target: [Array, Array, Array],
) -> Array:
    gd, state = nnx.split(model)
    unravel = jax.flatten_util.ravel_pytree(state)[1]

    def loss_fn(model: nnx.Module) -> Array:
        uvp = model(x[0])
        d2 = vmap(jax.hessian(model))(x[0])
        dd = jnp.trace(d2, axis1=2, axis2=3)
        d2u = dd[:, 0]
        d2v = dd[:, 1]
        gradup = vmap(jax.jacrev(model))(x[0])
        dpdx = gradup[:, 2, 0]
        dpdy = gradup[:, 2, 1]
        divu = gradup[:, 0, 0] + gradup[:, 1, 1]
        convx = gradup[:, 0, 0] * uvp[:, 0] + gradup[:, 0, 1] * uvp[:, 1]
        convy = gradup[:, 1, 0] * uvp[:, 0] + gradup[:, 1, 1] * uvp[:, 1]
        NSx = ((convx - nu * d2u + 1.0 / rho * dpdx) ** 2).mean()
        NSy = ((convy - nu * d2v + 1.0 / rho * dpdy) ** 2).mean()
        div = (divu**2).mean()

        # boundary
        uvpb = model(x[1])

        # pressure fixed in an arbitrary point
        p = model(x[2])[2]

        # return sum of losses
        return (
            NSx
            + NSy
            + div * 20
            + ((uvpb[:, :2] - target[1][:, :2]) ** 2).mean() * factor
            + (p**2).mean() * factor * 10
        )

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    loss_fn_split = lambda state: loss_fn(nnx.merge(gd, state))
    H_loss_fn = lambda flat_weights: loss_fn(nnx.merge(gd, unravel(flat_weights)))
    optimizer.update(
        gradients,
        grad=gradients,
        value_fn=loss_fn_split,
        value=loss,
        H_loss_fn=H_loss_fn,
    )
    return loss


t0 = time.time()


def run_optimizer(model, opt, num, name):
    for epoch in range(num):
        loss = train_step(
            model,
            opt,
            (t_data[0][epoch % batch], t_data[1][epoch % batch], t_data[2]),
            (y_data[0][epoch % batch], y_data[1][epoch % batch], y_data[2]),
        )
        if epoch % 100 == 0:
            print(f"Epoch {epoch} {name}, loss: {loss}")
        if abs(loss) < 1e-20:
            break


t0 = time.time()
run_optimizer(model, opt_adam, 50000, "Adam")
run_optimizer(model, opt_lbfgs, 100000, "LBFGS")
run_optimizer(model, opt_hess, 50000, "Hess")

yj = jnp.linspace(-1, 1, 50)
xx, yy = jnp.meshgrid(yj, yj, sparse=False, indexing="ij")
z = jnp.column_stack((xx.ravel(), yy.ravel()))
uvp = model(z)
gd, st = nnx.split(model)
pyt, ret = jax.flatten_util.ravel_pytree(st)


def compute_losses(model, x0, target0, batch):
    x = (x0[0][batch], x0[1][batch], x0[2])
    target = (target0[0][batch], target0[1][batch], target0[2])
    uvp = model(x[0])
    d2 = vmap(jax.hessian(model))(x[0])
    dd = jnp.trace(d2, axis1=2, axis2=3)
    d2u = dd[:, 0]
    d2v = dd[:, 1]
    gradup = vmap(jax.jacrev(model))(x[0])
    dpdx = gradup[:, 2, 0]
    dpdy = gradup[:, 2, 1]
    divu = gradup[:, 0, 0] + gradup[:, 1, 1]
    convx = gradup[:, 0, 0] * uvp[:, 0] + gradup[:, 0, 1] * uvp[:, 1]
    convy = gradup[:, 1, 0] * uvp[:, 0] + gradup[:, 1, 1] * uvp[:, 1]
    NSx = ((convx - nu * d2u + 1.0 / rho * dpdx) ** 2).mean()
    NSy = ((convy - nu * d2v + 1.0 / rho * dpdy) ** 2).mean()
    div = (divu**2).mean()

    # boundary
    uvpb = model(x[1])
    bnd = ((uvpb[:, :2] - target[1][:, :2]) ** 2).mean() * factor

    # pressure fixed in an arbitrary point
    p = model(x[2])[2].mean() ** 2 * factor

    return NSx, NSy, div, bnd, p


plt.contourf(xx, yy, uvp[:, 0].reshape(xx.shape))
plt.figure()
plt.contourf(xx, yy, uvp[:, 1].reshape(xx.shape))
plt.figure()
plt.contourf(xx, yy, uvp[:, 2].reshape(xx.shape))
plt.colorbar()
plt.show()
