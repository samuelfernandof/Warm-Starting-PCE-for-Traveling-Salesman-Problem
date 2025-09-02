import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import networkx as nx
import itertools
import numpy as np
import math
import pdb
import pulp


def sample_to_dict(sample, var_names):
    """
    Convert a Qiskit 'SolutionSample' to a {var_name: bit_value} dict.
    """
    # sample.x is an array/list of 0/1 in the same order as 'var_names'
    # We cast each bit to int just for safety.
    try:
        return {
            name: int(val)
            for name, val in zip(var_names, sample.x)
        }
    except:
        return {
            name: int(val)
            for name, val in zip(var_names, sample)
        }        


def interpret_solution(solution_dict, N):
    """
    Interpret a TSP solution with city 0 fixed at position 0 and 
    x_{i}_{p} = 1 meaning city i is placed at position p (1..N-1).
    
    Returns:
        tour (list): [0, city_for_position_1, city_for_position_2, ..., city_for_position_(N-1)]
        or None if invalid.
    """
    # We fix city 0 at position 0
    tour = [0] + [None]*(N-1)  # e.g. for N=4, tour = [0, None, None, None]
    
    # Counters to ensure each city & each position is used exactly once
    city_counts = [0]*N            # city_counts[i] => how many times city i appears
    position_counts = [0]*N        # position_counts[p] => how many times position p is occupied
    
    for i in range(1, N):
        for p in range(1, N):  # now p goes from 1..N-1 inclusive
            var_name_x_ip = f"x_{i}_{p}"
            if solution_dict.get(var_name_x_ip, 0) == 1:
                # Assign city i to position p
                city_counts[i] += 1
                position_counts[p] += 1
                
                # If city or position is already used more than once => invalid
                if city_counts[i] > 1 or position_counts[p] > 1:
                    return None
                
                # Place city i in the tour array at index p
                tour[p] = i
    
    # Check that each city i=1..(N-1) is used exactly once
    if any(city_counts[i] != 1 for i in range(1, N)):
        return None
    
    # Check that each position p=1..(N-1) is used exactly once
    if any(position_counts[p] != 1 for p in range(1, N)):
        return None
    
    return tour




def sample_and_plot_histogram_tsp(samples, adj_matrix, N, interpret_solution_fn,
                              top_n=30, var_names=None):
    """
    Interpret QUBO samples, validate solutions, and plot a histogram of the most sampled valid bitstrings.

    Parameters:
    - samples: Dictionary de {bitstring: frequency} vindo do Qiskit (ou outro sampler).
    - adj_matrix: Matriz de adjacência do grafo.
    - N: Número de nós do grafo.
    - Delta: Restrição de grau máximo (se aplicável).
    - interpret_solution_fn: Função que, dado um dicionário de variáveis -> valores, 
      retorne a solução interpretada (por exemplo, um conjunto de arestas MST).
    - top_n: Número de soluções mais comuns para mostrar no histograma.
    - var_names: Lista/ordem de variáveis, caso seja preciso mapear bits do bitstring.
    - v0: (opcional) se precisar excluir ou tratar um vértice específico, etc.

    Returns:
    - most_common_valid_solutions: Lista das soluções mais comuns (até `top_n`),
      onde cada item é (edges_solution, freq, [lista de bitstrings]).
    """
    
    # -------------------------------------------------------------------------
    # 1) Agregador para as soluções válidas: soma de frequências e bitstrings
    # -------------------------------------------------------------------------
    aggregated_solutions = defaultdict(lambda: {"freq": 0, "bitstrings": []})
    
    for bitstring, frequency in samples.items():
        # 1.1) Convertemos o bitstring em dicionário var->valor
        solution_dict = sample_to_dict(bitstring, var_names)
        converted_dict = {var.name: val for var, val in solution_dict.items()}
       

        # 1.2) Interpretamos a solução (por ex, extrair arestas do MST)
        tsp_solution = interpret_solution_fn(converted_dict, N)
        
        # Se a função interpretou e validou de fato (pode conter None se inválida)
        # 'tsp_solution' aqui deve ser algo como uma lista de edges (u,v,w)
        if tsp_solution and all(e is not None for e in tsp_solution):
            edges_tuple = tuple(tsp_solution)
            
            # 1.3) Acumule na nossa estrutura
            # Frequência multiplicada se você quiser "ampliar" a escala.
            # Usando frequency*10000 como no seu exemplo:
            freq_scaled = frequency * 10000
            aggregated_solutions[edges_tuple]["freq"] += freq_scaled
            aggregated_solutions[edges_tuple]["bitstrings"].append(bitstring)

    if not aggregated_solutions:
        print("No valid TSP solutions were found.")
        return []

    # -------------------------------------------------------------------------
    # 2) Ordenar as soluções por frequência (decrescente) e pegar top_n
    # -------------------------------------------------------------------------
    # aggregated_solutions.items() = [(edges_tuple, {"freq": X, "bitstrings": [...]})]
    sorted_agg = sorted(
        aggregated_solutions.items(),
        key=lambda item: item[1]["freq"],
        reverse=True
    )
    # Reduzimos aos top_n
    sorted_agg = sorted_agg[:top_n]

    # Montamos a lista final no formato que você quer exibir/devolver:
    # (edges_solution, freq, bitstrings)
    most_common_valid_solutions = [
        (edges_tuple, data["freq"], data["bitstrings"])
        for edges_tuple, data in sorted_agg
    ]

    # -------------------------------------------------------------------------
    # 3) Plotar histograma
    # -------------------------------------------------------------------------
    labels = [f"Solution {i+1}" for i in range(len(most_common_valid_solutions))]
    frequencies = [item[1] for item in most_common_valid_solutions]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, frequencies, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {top_n} Most Sampled Valid Solutions")
    plt.xlabel("Solutions")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()  # descomente se quiser exibir diretamente

    print("\nOs tours mais frequentes são:")
    for i, (tour, frequency, bitstring_list) in enumerate(most_common_valid_solutions, start=1):
        cost = sum(adj_matrix[tour[i], tour[(i + 1) % N]] for i in range(N))
        print(f"\nTour {i}: {tour}")
        print(f"Frequência: {frequency}")
        print(f"Custo total: {cost:.2f}")
        print("Bitstrings that produced this solution:")
        for bs in bitstring_list:
            print("  ", bs)
        print("-"*50)        

    # Interpretar o resultado
    most_common_tour, frequency, bitstring_list = most_common_valid_solutions[0]
    cost = sum(adj_matrix[most_common_tour[i], most_common_tour[(i + 1) % N]] for i in range(N))

    print("\nInterpretação do tour mais frequente:")
    print(f"O tour mais frequente encontrado pelo QAOA é {most_common_tour}, com um custo total de {cost:.2f}.")
    print("Este tour é a solução ótima encontrada para o problema do Caixeiro Viajante dado.")        

    return most_common_valid_solutions


def draw_tsp_solution(graph, solution_list, title="Traveling Salesman Solution"):
    """
    Visualize the TSP solution based on binary variable assignments such as x_{i}_{p} = 1.
    
    Parameters:
    - graph: An instance of your QAOA_TSP_Maxcut class, which has:
        - G: the underlying networkx Graph
        - n (or num_cities): number of cities
    - solution_dict: A dictionary { 'x_{i}_{p}': 0/1, ... } from Qiskit or from your aggregator.
    - title: Title of the plot.
    """
    #    Interpret the solution using the existing logic
    #    The interpret_solution function returns a list `tour = [0, cityA, cityB, ...]`
    #    or None if invalid. For example:
    #
    #    def interpret_solution(solution_dict, adj_matrix, N, Delta):
    #        ...
    #        return tour  # e.g. [0, 2, 3, 1], etc.
    #
    adj_matrix = nx.to_numpy_array(graph.G)  
    N = graph.num_nodes        
    tour = solution_list[0][0]    

    print(tour)
    
    # tour = interpret_solution(solution_dict, N)
    # if tour is None:
    #     print("Invalid TSP solution (it does not correspond to a valid route).")
    #     return
    
    # Build edges of the route
    route_edges = list(zip(tour, tour[1:]))
    # Let's close the loop:
    route_edges.append((tour[-1], tour[0]))
    
    # 2) Prepare a layout (or use a stored layout if you have one)
    pos = getattr(graph, 'pos', None)
    if pos is None:
        pos = nx.spring_layout(graph.G, seed=42)
    
    # Draw the graph
    plt.figure(figsize=(8, 6))
    
    # Draw all nodes
    nx.draw_networkx_nodes(graph.G, pos, node_color='lightblue', node_size=500)
    
    # Draw the TSP route edges in red
    nx.draw_networkx_edges(graph.G, pos, edgelist=route_edges, edge_color='red', width=2)
    
    # Draw the other edges (not used in the route) as dotted
    unused_edges = set(graph.G.edges()) - set(map(lambda e: tuple(sorted(e)), map(lambda x: tuple(sorted(x)), route_edges)))
    # Because (u,v) and (v,u) are the same in an undirected graph, handle them consistently:
    # We can unify them by always sorting each edge's tuple:
    G_edges_sorted = {tuple(sorted(e)) for e in graph.G.edges()}
    route_edges_sorted = {tuple(sorted(e)) for e in route_edges}
    dotted_edges = [tuple(e) for e in (G_edges_sorted - route_edges_sorted)]
    
    nx.draw_networkx_edges(graph.G, pos, edgelist=dotted_edges, style='dotted', edge_color='gray')
    
    # We'll label them as "city (pos p)"
    labels = {}
    for p, city in enumerate(tour):
        # e.g. "City 0 (pos 0)", "City 2 (pos 1)", ...
        labels[city] = f"{city} (pos {p})"
    
    nx.draw_networkx_labels(graph.G, pos, labels, font_size=12, font_color='black')
    
    plt.title(title)
    plt.axis("off")
    plt.show()


# def interpret_maxcut_solution(bitstring, G, variable_to_index = None):
#     """
#     Interpret a bitstring as a MaxCut partition.

#     Parameters:
#         bitstring (str): A string of '0's and '1's representing the partition (e.g., '10101').
#         G (nx.Graph): The graph with .edges(data='weight').

#     Returns:
#         cost (float): The total cut value.
#         cut_edges (list[tuple]): List of edges that cross the cut.
#         partition (set): Set of nodes in the partition (corresponding to '1's in the bitstring).
#     """
#     # Convert the bitstring into a set of nodes in the partition (corresponding to '1')
#     partition = {i for i, bit in enumerate(bitstring) if bit == '1'}

#     cut_edges = []
#     cost = 0.0

#     # For each weighted edge, check if it crosses the partition
#     for u, v, w in G.edges(data="weight", default=1):
#         if variable_to_index is not None:
#             u_idx = variable_to_index[u]  # Converte 'x_i_p' para um índice numérico
#             v_idx = variable_to_index[v] 
#         else:
#             u_idx = u           
#             v_idx = v           
#         if (u_idx in partition) != (v_idx in partition):  # crosses the partition
#             cost += w
#             cut_edges.append((u_idx, v_idx))

#     return cost, cut_edges, partition


def interpret_maxcut_solution(bitstring, G, variable_to_index = None, variavel_aux_fixa = True):
    """
    Interpreta uma bitstring como uma partição do problema de MaxCut.

    Parâmetros:
        bitstring (str): String de '0's e '1's representando a partição dos nós reais.
                         (A variável auxiliar não está incluída na bitstring, pois está fixada em 1.)
        G (nx.Graph): Grafo com arestas ponderadas. O nó auxiliar deve ter o nome 'aux'.

    Retorna:
        cost (float): Valor total do corte.
        cut_edges (list[tuple]): Lista de arestas que cruzam a partição.
        partition (set): Conjunto de índices dos nós reais que estão no lado "1" da partição.
    """
    # Converte a bitstring para um conjunto de índices correspondentes aos nós reais (0 a n-1)
    partition = {i for i, bit in enumerate(bitstring) if bit == '1'}

    cut_edges = []
    cost = 0.0

    # Percorre todas as arestas ponderadas do grafo
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        # Caso a aresta envolva o nó auxiliar:
        if u == 'aux' and v != 'aux' and variavel_aux_fixa:
            # Como o nó auxiliar está fixo em 1, a aresta (aux, v) está no corte se v tiver valor 0.
            # Obtemos o índice de v (supondo que variable_to_index esteja disponível)
            v_idx = variable_to_index[v] 
            if v_idx not in partition:  # v = 0
                cost += w
                cut_edges.append((u, v))
        elif v == 'aux' and u != 'aux' and variavel_aux_fixa:
            u_idx = variable_to_index[u] 
            if u_idx not in partition:  # u = 0
                cost += w
                cut_edges.append((u, v))
        else:
            # Aresta entre nós reais: a aresta está no corte se os nós estiverem em partições diferentes
            u_idx = variable_to_index[u] 
            v_idx = variable_to_index[v] 
            if (u_idx in partition) != (v_idx in partition):  # XOR
                cost += w
                cut_edges.append((u, v))

    return cost, cut_edges, partition

def sample_and_plot_maxcut_histogram(samples, G, top_n=10):
    """
    Interpret MaxCut QAOA samples, compute cut values, and plot a histogram.

    Parameters:
        samples (dict): A dictionary {bitstring: frequency}.
        G (nx.Graph): The graph for MaxCut.
        top_n (int): Number of top solutions to display.

    Returns:
        most_common_solutions: List of (cost, frequency, bitstring, cut_edges).
    """
    # Store solutions with their computed costs and frequencies
    solution_data = []

    for bitstring, freq in samples.items():
        # Interpret each bitstring
        cost, cut_edges, partition = interpret_maxcut_solution(bitstring, G)
        solution_data.append((cost, freq, bitstring, cut_edges))

    # Sort by cost (descending) and frequency (descending)
    sorted_solutions = sorted(solution_data, key=lambda x: (-x[1]))

    # Select the top_n solutions
    top_solutions = sorted_solutions[:top_n]

    # Prepare data for the histogram
    labels = [f"{bitstring}" for _,_,bitstring,_ in top_solutions]
    frequencies = [data[1] for data in top_solutions]

    # Plot the histogram
    plt.figure(figsize=(8, 5))
    plt.bar(labels, frequencies, color="skyblue")
    plt.title(f"Top {top_n} Most Frequent MaxCut Solutions")
    plt.xlabel("Cut #")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Print details of the top solutions
    print("\nTop MaxCut Solutions:")
    for i, (cost, freq, bitstring, cut_edges) in enumerate(top_solutions, start=1):
        print(f"Solution {i}:")
        print(f"  Bitstring: {bitstring}")
        print(f"  Cost: {cost}")
        print(f"  Frequency: {freq}")
        print(f"  Cut Edges: {cut_edges}")
        print("-" * 50)

    return top_solutions


def draw_maxcut_solution(G, partition, cut_edges, title="MaxCut Solution"):
    """
    Draws the graph with two colors for the two partitions, 
    and highlights the cut edges.
    
    partition: set of nodes that are in side A (others are in side B)
    cut_edges: list of edges that cross the cut
    """
    pos = nx.spring_layout(G, seed=42)  # or any layout
    color_map = []
    for node in G.nodes():
        if node in partition:
            color_map.append('red')
        else:
            color_map.append('blue')
    
    plt.figure(figsize=(8,6))
    
    # Draw all edges first with a lighter style
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Highlight cut edges
    nx.draw_networkx_edges(
        G, pos, edgelist=cut_edges, edge_color='green', width=2
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=600)
    nx.draw_networkx_labels(G, pos, font_color='white')
    
    plt.title(title)
    plt.axis('off')
    plt.show()


def pick_and_draw_a_top_solution(G, best_bitstring):

    cost, cut_edges, partition = interpret_maxcut_solution(best_bitstring, G)

    # Directly derive the partition from the bitstring
    partition = {i for i, bit in enumerate(best_bitstring) if bit == '1'}
    
    print(f"Best Bitstring: {best_bitstring}")
    print(f"Partition: {partition}")
    print(f"Cut Edges: {cut_edges}")
    print(f"Cost: {cost}")


    # Draw the solution
    draw_maxcut_solution(G, partition, cut_edges, title="Top MaxCut QAOA Solution")

def interpret_solution_tsp(qubit_index, bitstring, N):
    # A cidade 0 está fixada na posição 0, então já podemos inicializar a turnê com isso
    tour = [0] + [None] * (N - 1)  # A cidade 0 já está na posição 0
    city_counts = [0] * (N - 1)  # Contador para as cidades (0 a N-1)
    position_counts = [0] * (N - 1)  # Contador para as posições (0 a N-1)

    # print(bitstring)
    
    # Iterando sobre as cidades de 1 até N-1
    for i in range(1, N):
        for p in range(1, N):  # Posições de 1 a N-1
            # idx = (i-1) * (N-1) + p  # Novo índice com a redução (eliminando cidade 0)
            if bitstring[qubit_index['x', i, p]] == '1':
                # print(qubit_index['x', i, p])
                # print('i:', i)
                # print('p:', p)
                city_counts[i-1] += 1
                position_counts[p-1] += 1
                # Verificar se alguma cidade ou posição foi atribuída mais de uma vez
                if city_counts[i-1] > 1 or position_counts[p-1] > 1:
                    # Solução inválida
                    # if city_counts[i] > 1:
                    #     print(f"Erro: Cidade {i} atribuída mais de uma vez.")
                    # if position_counts[p] > 1:
                    #     print(f"Erro: Posição {p} atribuída mais de uma vez.")                        
                    return None
                tour[p] = i  # Posição p, porque p começa em 1, já que p=0 é ocupado pela cidade 0

    # Verificando se todas as cidades (exceto cidade 0) aparecem uma vez
    if any(count != 1 for count in city_counts):
        return None
    # Verificando se todas as posições (1 a N-1) são preenchidas
    if any(count != 1 for count in position_counts):
        return None

    # Se tudo estiver correto, retorna a turnê
    return tour


def solve_maxcut_exact(graph: nx.Graph, variavel_aux_fixa=True):
    """
    Resolve exatamente o problema de MaxCut via ILP usando PuLP, 
    fixando o nó auxiliar (por exemplo, 'aux') em 1.
    
    Parâmetros:
        graph (nx.Graph): Um grafo do NetworkX. Pode ser ponderado ou não.
                          Se não ponderado, o atributo 'weight' de cada aresta assume 1.
    
    Retorna:
        (set_s, set_t, max_cut_value)
            set_s: Lista de nós na partição 0.
            set_t: Lista de nós na partição 1 (incluindo o nó auxiliar 'aux', que é fixo em 1).
            max_cut_value: Valor ótimo do corte.
    """
    import pulp
    
    # 1) Cria o problema de maximização
    prob = pulp.LpProblem("MaxCut", pulp.LpMaximize)
    
    # 2) Cria variáveis binárias para os nós reais (excluindo o nó auxiliar)
    if variavel_aux_fixa:
        nodes = [i for i in graph.nodes() if i != 'aux']
    else:
        nodes = [i for i in graph.nodes()]
        
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in nodes}
    # O nó auxiliar é fixo em 1 (não há variável associada a ele)
    
    # 3) Cria variáveis binárias para as arestas entre nós reais (para modelar o produto x_i * x_j)
    y = {}
    for (i, j) in graph.edges():
        if (i == 'aux' or j == 'aux') and variavel_aux_fixa:
            continue
        # Para evitar duplicação, consideramos apenas i < j
        if i < j:
            y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", cat=pulp.LpBinary)
        else:
            y[(j, i)] = pulp.LpVariable(f"y_{j}_{i}", cat=pulp.LpBinary)
    
    # 4) Adiciona restrições para garantir que y_{ij} = x_i * x_j para arestas entre nós reais
    for (i, j) in y:
        prob += y[(i, j)] >= x[i] + x[j] - 1
        prob += y[(i, j)] <= x[i]
        prob += y[(i, j)] <= x[j]
    
    # 5) Define a função objetivo
    # Para arestas entre nós reais, a contribuição é: w_{ij}*(x_i + x_j - 2*x_i*x_j)
    objective_terms = []
    for (i, j) in y:
        w_ij = graph[i][j].get('weight', 1.0)
        objective_terms.append(w_ij * (x[i] + x[j] - 2*y[(i, j)]))
    
    # Para as arestas que ligam o nó auxiliar aos nós reais:
    # Como o nó auxiliar está fixo em 1, para uma aresta (aux, i) de peso w a contribuição é:
    #   w*(1 + x_i - 2*1*x_i) = w*(1 - x_i)
    if variavel_aux_fixa:
        for (u, v, data) in graph.edges(data=True):
            if u == 'aux' and v != 'aux':
                w_uv = data.get('weight', 1.0)
                objective_terms.append(w_uv * (1 - x[v]))
            elif v == 'aux' and u != 'aux':
                w_uv = data.get('weight', 1.0)
                objective_terms.append(w_uv * (1 - x[u]))
    
    prob += pulp.lpSum(objective_terms), "MaxCutObjective"
    
    # 6) Resolve o problema ILP usando o solver CBC do PuLP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 7) Recupera o valor ótimo do corte
    max_cut_value = pulp.value(prob.objective)
    
    # 8) Reconstrói os conjuntos de corte:
    #    - Para os nós reais: se x_i < 0.5, atribui ao conjunto S; se x_i >= 0.5, ao conjunto T.
    #    - Adiciona o nó auxiliar em T (pois está fixo em 1).
    set_s = []
    set_t = []
    for i in nodes:
        if pulp.value(x[i]) < 0.5:
            set_s.append(i)
        else:
            set_t.append(i)
    if variavel_aux_fixa:
        set_t.append('aux')
    
    return set_s, set_t, max_cut_value

# def solve_maxcut_exact(graph: nx.Graph):
#     """
#     Solve MaxCut exactly via ILP using PuLP.

#     :param graph: A networkx Graph. Can be weighted or unweighted.
#                   If unweighted, edge attribute 'weight' defaults to 1.
#     :return: (set_s, set_t, max_cut_value)
#              set_s  = list of nodes in partition 0
#              set_t  = list of nodes in partition 1
#              max_cut_value = value of the cut
#     """

#     # Create a Maximize problem
#     prob = pulp.LpProblem("MaxCut", pulp.LpMaximize)

#     # 1) Create binary variables x_i for each node
#     x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in graph.nodes()}

#     # 2) Create binary variables y_{ij} for each edge (i<j to avoid duplication)
#     y = {}
#     for (i, j) in graph.edges():
#         if i < j:
#             y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", cat=pulp.LpBinary)
#         else:
#             y[(j, i)] = pulp.LpVariable(f"y_{j}_{i}", cat=pulp.LpBinary)

#     # 3) Add constraints to link y_{ij} = x_i * x_j
#     for (i, j) in y:
#         prob += y[(i, j)] >= x[i] + x[j] - 1
#         prob += y[(i, j)] <= x[i]
#         prob += y[(i, j)] <= x[j]
#         # y[(i, j)] >= 0 is implicitly satisfied since y is binary

#     # 4) Define the objective function
#     #    For unweighted edges, the default edge weight = 1 if not provided.
#     objective_terms = []
#     for (i, j) in y:
#         w_ij = graph[i][j].get('weight', 1.0)
#         # contribution = w_ij * (x_i + x_j - 2y_ij)
#         objective_terms.append(w_ij * (x[i] + x[j] - 2*y[(i, j)]))
    
#     prob += pulp.lpSum(objective_terms), "MaxCutObjective"

#     # 5) Solve the ILP
#     prob.solve(pulp.PULP_CBC_CMD(msg=False))  # use default CBC solver

#     # 6) Retrieve the optimal objective value (max cut)
#     max_cut_value = pulp.value(prob.objective)

#     # 7) Reconstruct the cut sets
#     set_s = []
#     set_t = []
#     for i in graph.nodes():
#         # If x_i is 0, we put i in set_s; if 1, in set_t
#         if pulp.value(x[i]) < 0.5:
#             set_s.append(i)
#         else:
#             set_t.append(i)

#     return set_s, set_t, max_cut_value


def draw_maxcut_graph(G_maxcut):
    """ Desenha o grafo MaxCut, mostrando os rótulos das variáveis explicitamente """

    plt.figure(figsize=(10, 7))
    
    # Layout para organização visual
    pos = nx.spring_layout(G_maxcut, seed=42)  # Usa layout de força para distribuir os nós
    
    # Destacar o nó auxiliar se existir
    aux_node = "aux" if "aux" in G_maxcut.nodes else None

    # Definir cores diferentes para o nó auxiliar e as variáveis
    node_colors = ["red" if node == aux_node else "lightblue" for node in G_maxcut.nodes]

    # Desenhar os nós
    nx.draw(G_maxcut, pos, with_labels=True, node_color=node_colors, node_size=2000, font_size=10, edge_color="gray")
    
    # Adicionar os pesos das arestas como rótulos
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G_maxcut.edges(data=True)}
    nx.draw_networkx_edge_labels(G_maxcut, pos, edge_labels=edge_labels, font_size=8)
    
    # Título e exibição
    plt.title("Grafo MaxCut para o Problema TSP (QUBO -> MaxCut)", fontsize=12)
    plt.show()
