from graph import Graph
from qubo_problem import VQE_Multibasis
from plot_solutions import (draw_tsp_solution, sample_and_plot_histogram_tsp, 
    pick_and_draw_a_top_solution, interpret_solution, sample_and_plot_maxcut_histogram, 
    draw_maxcut_graph, solve_maxcut_exact)
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA
from config import Config
import numpy as np
import csv

def main():
    config = Config()
    N = 5
    epsilon = 0.2
    weight_range = (10, 100)
    # for erro in [0.01, 0.05]:
    for seed in range(40, 51):
    # seed = 50
        TSP = True

        graph = Graph(N, weight_range, seed)
        # graph.draw()

        # Generate the adjacency matrix
        adj_matrix = np.zeros((N, N), dtype=int)
        for (u, v, data) in graph.G.edges(data=True):
            adj_matrix[u][v] = data['weight']
            adj_matrix[v][u] = data['weight']  # Ensure symmetry       

        # Solve TSP using brute force
        if TSP:
            route, distance = graph.brute_force_tsp(adj_matrix)
            print("Shortest Route:", route)
            print("Total Distance:", distance)
            max_edge_weight = max(data['weight'] for _, _, data in graph.G.edges(data=True))
            print("Max Weight of Graph:", max_edge_weight)   
            # graph.plot_tsp_route(route)

        else:
            correct_cost,_,_ = graph.brute_force_maxcut()
            graph.draw_brute_force_maxcut_solution()    

        k=2
        # 1) Cria TSP instance
        tsp_model = VQE_Multibasis(graph.G, config, k=k, pauli_ops = ["Z", "X", "Y"], TSP=True, fake_backend=True, seed=seed)
        # 2) Converte TSP -> QUBO -> MaxCut
        graph_tsp_to_maxcut, variables = tsp_model.convert_tsp_to_maxcut()
        # draw_maxcut_graph(graph_tsp_to_maxcut)

        # Declaring variables  
        c_stars = None
        variavel_aux_fixa = False

        # Solve TSP via MaxCut para 100 seeds
        for seed_sample in range(1, 11):
            # 3) Cria segunda instância (MaxCut) mas usando as variáveis do TSP
            qubo_problem_maxcut = VQE_Multibasis(
                G=graph_tsp_to_maxcut, 
                config=config, 
                k=k, 
                pauli_ops = ["Z", "X", "Y"],
                TSP=False,           # MaxCut
                fake_backend=False, 
                seed=seed_sample,
                warm_starting=True,   # warm-starting=True e epsilon=0.5 equivale, obviamente, a warm-starting=False
                epsilon = epsilon,
                external_var_names=variables,  # <--- ESSENCIAL
                variable_aux_fixed=variavel_aux_fixa
            )

            p = 1
            bitstring, old_cost, new_bitstring, new_cost, bitstring_GW, cut_value_GW, c_stars = qubo_problem_maxcut.solve_problem(p)

            # Solve exato do MaxCut para fins de comparação
            _, _, max_cut_value = solve_maxcut_exact(graph_tsp_to_maxcut, variavel_aux_fixa)
            
            # 4) Reconstrói rota TSP usando o 1o model
            tour, distance_result, flipped_bitstring = tsp_model.maxcut_solution_to_tsp(new_bitstring, variables)

            print(f"seed_sample={seed_sample}, final_bitstring={bitstring}, improved={new_bitstring}")
            print(f"MaxCut value = {new_cost} / {max_cut_value}")
            print("tour =>", tour)

            # Salvar CSV
            with open(f"tsp_to_maxcut_N_{N}_k_{k}_p{p}_{epsilon}_fakebackend.csv", "a", newline="") as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow([
                        "seed_grafo", "N", "num_layer","epsilon", 
                        "final_bitstring", "final_cost", 
                        "final_bitstring_pos_classical",  "bitstring_tsp", "cost_pos_classical",
                        "cut_value_GW", "bitstring_GW",
                        "correct_cost", "ratio", 
                        "tour", "cost_tsp_result", "true_tour", "true_cost_tsp"
                    ])
                ratio = new_cost / max_cut_value if max_cut_value != 0 else 0
                writer.writerow([
                    seed, N, p, epsilon,
                    bitstring, old_cost,
                    new_bitstring, flipped_bitstring, new_cost,
                    cut_value_GW, bitstring_GW,
                    max_cut_value, ratio,
                    tour, distance_result, route, distance,
                ])


if __name__ == "__main__":
    main()
