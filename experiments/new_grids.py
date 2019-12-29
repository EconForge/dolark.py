from dolark.dolo_improvements import multimethod, get_coefficients, eval_is, eval_ms, Linear, EmptyGrid, WarpGrid, CartesianGrid, UnstructuredGrid, DecisionRule, Cubic
import numpy as np

g = CartesianGrid([0.0,0], [1.0,1.0], [10,10])
wg = WarpGrid(g, ['exp','exp'])


from matplotlib import pyplot as plt
nn = wg.nodes()

plt.plot(nn[:,0], nn[:,1],'o')
for i in range(wg.n_nodes()):
    nnn = wg.node(i)
    plt.plot(nnn[0], nnn[1], 'x', color='red')


nn = wg.nodes()
exo_grid = UnstructuredGrid(np.array([[0.2, 0.5, 0.7]]))



f = lambda x,y: x**2 + y
vals = f(nn[:,0], nn[:,1]).reshape((1,10,10,1))
vals



fg = CartesianGrid([1.0,1.0], [2.7,2.7], [1000,1000])
no = fg.nodes()
tvals = f(no[:,0], no[:,1])




dr = DecisionRule(exo_grid, wg, 'linear')
dr.set_values(vals)

s = wg.nodes()
x = dr.eval_is(0,s)
assert( abs(x.ravel() - vals.ravel()).max()<1e-10 )

abs(dr.eval_is(0,no).ravel() - tvals.ravel()).max()


# cubic approximation is better

dr = DecisionRule(exo_grid, wg, 'cubic')
dr.set_values(vals)

s = wg.nodes()
x = dr.eval_is(0,s)
assert( abs(x.ravel() - vals.ravel()).max()<1e-10 )

abs(dr.eval_is(0,no).ravel() - tvals.ravel()).max()


## let's check extrapolation properties


g = WarpGrid( CartesianGrid([0.0], [1.0], [10]), ['exp'] )
f = lambda x: np.exp(x[:,0])[:,None]
nodes = g.nodes()
vals = f(nodes).reshape((1,10,1))


dr = DecisionRule(exo_grid, g, 'linear')
dr.set_values(vals)


fg = CartesianGrid([0.2], [3.0], [1000])
nn = fg.nodes()

tvals = f(nn)
xx = dr.eval_is(0, nn)

plt.figure(figsize=(15,10))
plt.plot(nodes.ravel(), vals.ravel(), 'o', label='data')
plt.plot(nn.ravel(), xx.ravel(), label='true')
plt.plot(nn.ravel(), tvals.ravel(), label='warped')
plt.grid()



## let's check exp(exp)


g = WarpGrid( CartesianGrid([-5], [0.5], [10]), ['exp(exp)'] )
f = lambda x: (x[:,0]**2)[:,None]
nodes = g.nodes()
vals = f(nodes).reshape((1,10,1))


dr = DecisionRule(exo_grid, g, 'linear')
dr.set_values(vals)

fg = CartesianGrid([0], [2], [1000])
nn = fg.nodes()

tvals = f(nn)
xx = dr.eval_is(0, nn)

plt.figure(figsize=(15,10))
plt.plot(nodes.ravel(), vals.ravel(), 'o', label='data')
plt.plot(nn.ravel(), xx.ravel(), label='true')
plt.plot(nn.ravel(), tvals.ravel(), label='warped')
# plt.xscale('log')
# plt.yscale('log')
plt.ylim(0,5)
plt.xlim(0,2)
plt.grid()
# sounds like a good idea, but no thanks.


N = 10

a = np.exp(np.exp( np.linspace(-10,1, N) ))
b = np.linspace(0,1)

from dolark.dolo_improvements import ICartesianGrid


g = ICartesianGrid([a])
f = lambda x: (x[:,0]**2)[:,None]
nodes = g.nodes()
vals = f(nodes).reshape((1,N,1))


dr = DecisionRule(exo_grid, g, 'linear')
dr.set_values(vals)

fg = CartesianGrid([0], [20], [1000])
nn = fg.nodes()

tvals = f(nn)
xx = dr.eval_is(0, nn)

plt.figure(figsize=(15,10))
plt.plot(nodes.ravel(), vals.ravel(), 'o', label='data')
plt.plot(nn.ravel(), xx.ravel(), label='true')
plt.plot(nn.ravel(), tvals.ravel(), label='warped')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(0,5)
# plt.xlim(0,2)
plt.grid()
# sounds like a good idea, but no thanks.


# 2d

g = ICartesianGrid([a])
f = lambda x: (x[:,0]**2)[:,None]
nodes = g.nodes()
vals = f(nodes).reshape((1,N,1))


dr = DecisionRule(exo_grid, g, 'linear')
dr.set_values(vals)

fg = CartesianGrid([0], [20], [1000])
nn = fg.nodes()

tvals = f(nn)
xx = dr.eval_is(0, nn)

plt.figure(figsize=(15,10))
plt.plot(nodes.ravel(), vals.ravel(), 'o', label='data')
plt.plot(nn.ravel(), xx.ravel(), label='true')
plt.plot(nn.ravel(), tvals.ravel(), label='warped')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(0,5)
# plt.xlim(0,2)
plt.grid()
# sounds like a good idea, but no thanks.
