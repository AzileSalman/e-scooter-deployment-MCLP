import pulp

def build_mingpds_model(demand_points, facility_sites, coverage_sets, weights, K):
    # Defining the  ILP model
    model = pulp.LpProblem("MinGPDS", pulp.LpMinimize)
    
    # Decision variables
    x = pulp.LpVariable.dicts("x", facility_sites, lowBound=0, upBound=1, cat="Binary")
    y = pulp.LpVariable.dicts("y", demand_points, lowBound=0, upBound=1, cat="Binary")
    
    # the Objective: minimize number of facilities
    model += pulp.lpSum(x[j] for j in facility_sites)

    # coverage constraint
    for i in demand_points:
        if coverage_sets[i]:
            model += y[i] <= pulp.lpSum(x[j] for j in coverage_sets[i])
        else:
            model += y[i] == 0
    # Profit constraint
    model += pulp.lpSum(weights[i] * y[i] for i in demand_points) >= K, "ProfitThreshold"
    
    # Solving
    model.solve(pulp.PULP_CBC_CMD(msg=1))  # msg=1 to see solver logs
    
    # Extracting the results
    selected_facilities = [j for j in facility_sites if pulp.value(x[j]) == 1]
    covered_nodes = [i for i in demand_points if pulp.value(y[i]) == 1]
    total_covered = sum(weights[i] for i in covered_nodes)
    
    return model, selected_facilities, covered_nodes, total_covered
