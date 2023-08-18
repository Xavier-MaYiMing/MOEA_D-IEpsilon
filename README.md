### MOEA/D-IEpsilon: Multi-objective evolutionary algorithm based on decomposition and improved epsilon constraint-handling mechanism

##### Reference: Fan Z, Li W, Cai X, et al. An improved epsilon constraint-handling method in MOEA/D for CMOPs with large infeasible regions[J]. Soft Computing, 2019, 23: 12491-12510.

##### MOEA/D-IEpsilon is a constrained multi-objective evolutionary algorithm (CMOEA). The epsilon level is adjust according to the ratio of feasible to total solutions in the current population.

| Variables | Meaning                                                      |
| --------- | ------------------------------------------------------------ |
| npop      | Population size                                              |
| iter      | Iteration number                                             |
| lb        | Lower bound                                                  |
| ub        | Upper bound                                                  |
| T         | Neighborhood size (default = 30)                             |
| delta     | The probability of selecting individuals in the neighborhood (default = 0.9) |
| nr        | The maximal number of solutions replaced by a child (default = 2) |
| tau       | Control the scale factor multiplied by the maximum overall constraint violation (default = 0.1) |
| alpha     | Control the searching preference between the feasible and infeasible regions (default = 0.95) |
| Tc        | Control generation (default = 0.8 * iter)                    |
| CR        | Crossover rate (default = 1)                                 |
| F         | Mutation scalar number (default = 0.5)                       |
| pm        | Mutation probability (default = 1)                           |
| eta_m     | Spread factor distribution index (default = 20)              |
| nvar      | The dimension of decision space                              |
| nobj      | The dimension of objective space                             |
| V         | Weight vectors                                               |
| B         | The T closet weight vectors                                  |
| pop       | Population                                                   |
| objs      | Objectives                                                   |
| phi       | Constraint violations                                        |
| phi_max   | The maximum overall constraint violation found so far        |
| rk        | The ratio of feasible solutions to all solutions             |
| z         | Ideal point                                                  |
| NS        | Feasible non-dominated solutions                             |
| rp        | Random permutation                                           |
| off       | Offspring                                                    |
| off_obj   | Offspring objecrtive                                         |
| off_phi   | Offspring constraint violation                               |
| c         | Update counter                                               |

#### Test problem: LIR-CMOP6

$$
\begin{aligned}
& J_1=\{3, 5, \cdots, 29\}, J_2 = \{2, 4, \cdots, 30\} \\
& g_1(x) = \sum_{i \in J_1} (x_i - \sin(0.5i\pi x_1/30))^2 \\
& g_2(x) = \sum_{i \in J_2} (x_i - \cos(0.5i\pi x_1/30))^2 \\
&\min \\
& f_1(x) = x_1 + 10g_1(x) + 0.7057 \\
& f_2(x) = 1 - x_1^2 + 10g_2(x) + 0.7057 \\
& \text{subject to} \\
& c_k(x) = ((f_1 - p_k)\cos\theta - (f_2 - q_k)\sin\theta)^2/a_k^2 + ((f_1 - p_k)\sin\theta - (f_2 - q_k)\cos\theta)^2/b_k^2 \geq r \\
& p_k = [1.8, 2.8], \quad q_k = [1.8, 2.8], \quad a_k = [2, 2], \quad b_k = [8, 8], \quad k=1, 2 \\
& r = 0.1, \quad \theta = -0.25 \pi \\
& x_i \in [0, 1], \quad i = 1, \cdots, 30
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(300, 1000, np.array([0] * 30), np.array([1] * 30))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/MOEA_D-IEpsilon/blob/main/Pareto%20front.png)



