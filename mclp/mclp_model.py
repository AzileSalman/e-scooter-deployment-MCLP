import pulp

def build_mclp_model(demand_points, facility_sites, coverage_sets, weights, p):
    x = {j: pulp.LpVariable(f"x_{j}", cat="Binary") for j in facility_sites}
    y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in demand_points}

    model = pulp.LpProblem("MCLP", pulp.LpMaximize)
    model += pulp.lpSum(weights[i] * y[i] for i in demand_points)

    for i in demand_points:
        if coverage_sets[i]:
            model += y[i] <= pulp.lpSum(x[j] for j in coverage_sets[i])
        else:
            model += y[i] == 0

    model += pulp.lpSum(x[j] for j in facility_sites) <= p

    model.solve()

    selected = [j for j in facility_sites if x[j].value() == 1]
    covered = [i for i in demand_points if y[i].value() == 1]
    total_covered = sum(weights[i] for i in covered)

    return model, selected, covered, total_covered
