import openturns as ot

# OpenTURNS nonlinear least-squares calibration reference family:
# y = a + b * exp(c * x)
model = ot.SymbolicFunction(
    ["a", "b", "c", "x"],
    ["a + b * exp(c * x)"],
)
model.setOutputDescription(["y"])

# These distributions make the four-input model independently valid as f(x).
# Calibration Studio treats a, b, and c as fixed parameters and conditions on
# observed x values instead of sampling this project distribution.
a = ot.Uniform(0.0, 5.0)
a.setDescription(["a"])

b = ot.Uniform(0.5, 2.0)
b.setDescription(["b"])

c = ot.Uniform(0.1, 0.6)
c.setDescription(["c"])

x = ot.Uniform(0.5, 9.5)
x.setDescription(["x"])

problem = ot.JointDistribution([a, b, c, x])
