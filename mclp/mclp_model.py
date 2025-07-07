import pulp

def build_mclp_model(demand_points, facility_sites, coverage_sets, weights, p):
    # binary decision variable indicating whether a facility is placed at location j
    x = {j: pulp.LpVariable(f"x_{j}", cat="Binary") for j in facility_sites}
    # binary decision variable indicating whether demand point i is served
    y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in demand_points}
    # defining LP for MCLP for maximizing
    model = pulp.LpProblem("MCLP", pulp.LpMaximize)
    # objective
    model += pulp.lpSum(weights[i] * y[i] for i in demand_points)
    # coverage constraint
    for i in demand_points:
        if coverage_sets[i]:
            model += y[i] <= pulp.lpSum(x[j] for j in coverage_sets[i])
        else:
            model += y[i] == 0
    # facility limit constraint
    model += pulp.lpSum(x[j] for j in facility_sites) <= p
    # solving model
    model.solve()
    # extracting list of selected facilities (solution)
    selected = [j for j in facility_sites if x[j].value() == 1]
    # extracting list of covered demand points
    covered = [i for i in demand_points if y[i].value() == 1]
    # total weight of covered demand
    total_covered = sum(weights[i] for i in covered)

    return model, selected, covered, total_covered
