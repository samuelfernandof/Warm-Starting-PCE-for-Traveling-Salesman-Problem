from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_algorithms import SamplingVQE

from qiskit.primitives import BackendSampler
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer import AerSimulator
from qiskit_optimization.problems.variable import VarType
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler as RuntimeSampler
# from qiskit_ibm_runtime import SamplerV2 as Sampler
from scipy.optimize import minimize
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.noise import phase_damping_error
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane



from qiskit_optimization.algorithms import GoemansWilliamsonOptimizer

import cvxpy as cp
from scipy.linalg import eigh


import numpy as np
import time
import networkx as nx
import itertools
import math
from plot_solutions import interpret_maxcut_solution, interpret_solution_tsp
from collections import Counter
import copy

from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit_algorithms.optimizers import ADAM


class VQE_Multibasis:
    def __init__(
        self,
        G,
        config,
        seed=42,
        TSP=True,
        A=400,
        k=2,
        pauli_ops = ["Z", "X", "Y"],
        fake_backend=False,
        warm_starting = False,
        epsilon=0.25,
        external_var_names=None,
        c_stars = None,
        variable_aux_fixed = True,
        mean_GW = 0,
        num_cuts = 100
    ):
        """
        Se TSP=True, cria as variáveis x_{i}_{p} e define objetivo TSP.
        Se TSP=False e external_var_names=None, cria x{0}, x{1}, ... (MaxCut "padrão").
        Se TSP=False e external_var_names != None, PULA a criação local das variáveis
          e usa external_var_names (vindas do TSP->QUBO->MaxCut).
          Se TSP=False e external_var_names != None e multibasis = True, pula também a criação da função objetivo
        """
        self.G = G
        self.config = config
        self.seed = seed
        self.TSP = TSP
        self.k = k
        self.pauli_ops = pauli_ops
        self.A = A
        self.fake_backend = fake_backend
        self.warm_starting = warm_starting
        self.variable_aux_fixed = variable_aux_fixed
        self.c_stars = c_stars
        self.cut_value_GW = 0
        self.mean_GW = mean_GW
        self.num_cuts = num_cuts


        self.epsilon = epsilon

        # Obter número de nós do grafo
        self.n = G.number_of_nodes()

        # Se TSP, m = (n-1)^2, senão m = n
        if self.TSP:
            self.Maxcut = False
            self.m = (self.n - 1) ** 2
        else:
            self.Maxcut = True
            if external_var_names is None:
                self.m = self.n
            else:
                if self.variable_aux_fixed:
                    self.m = self.n - 1
                else:
                    self.m = self.n


        # Cria matrix adj e dictionary
        self.variable_to_index = {var: idx for idx, var in enumerate(self.G.nodes)}

        self.adj_matrix = np.zeros((self.n, self.n), dtype=int)
        for (u, v, data) in self.G.edges(data=True):
            u_idx = self.variable_to_index[u]
            v_idx = self.variable_to_index[v]
            self.adj_matrix[u_idx][v_idx] = data["weight"]
            self.adj_matrix[v_idx][u_idx] = data["weight"]

        print(self.adj_matrix)

        self.nu = np.sum(np.triu(self.adj_matrix, k=1))

        print("Número total de qubits necessários (m):", self.m)    

        self.qb = QuadraticProgram()
        
        if self.Maxcut and external_var_names is not None:
            # 1) Use exatamente os nomes vindos do TSP->QUBO->MaxCut
            self.var_names = list(external_var_names)

            # 2) Crie as variáveis do QP com estes nomes (não crie 'aux' se for fixo)
            for name in self.var_names:
                if self.variable_aux_fixed and name == "aux":
                    continue
                self.qb.binary_var(name)

            # (opcional) se NÃO for fixar o aux, garanta que ele exista no QP:
            if (not self.variable_aux_fixed) and ("aux" in self.G.nodes):
                if "aux" not in [v.name for v in self.qb.variables]:
                    self.qb.binary_var("aux")

            # 3) Monte objetivo a partir do grafo de MaxCut (em vez de _define_objective_function)
            self._define_objective_from_maxcut_graph()   # <- NOVA FUNÇÃO (abaixo)
            print("\n[INFO] MaxCut from TSP: using EXTERNAL var names and objective from G_maxcut.\n")

        else:
            # Fluxo padrão
            self._configure_variables()
            self._define_objective_function()
            if self.TSP:
                self._add_constraints()


        # Setup da codificação multibasis
        self.pauli_strings, self.n_qubits = self.multibasis_encoding_multiplets_qiskit(self.m, self.k)
        print("Pauli strings (multibasis):", self.pauli_strings)        
        print("n_qubits (multibasis):", self.n_qubits)
        self.alpha = 1.5 * self.n_qubits
        self.beta = 0.5

        self.objective_func_vals = []


    def _configure_variables(self):
        # Se self.Maxcut=True e self.TSP=False, criamos x{i_idx}
        # Se TSP=True, criamos x_{i}_{p}
        if self.Maxcut and (not self.TSP):
            # Criar x0, x1, x2... para cada nó
            for i in self.G.nodes:
                i_idx = self.variable_to_index[i]
                self.qb.binary_var(f"x{i_idx}")
            self.var_names = self.qb.variables

        elif self.TSP:
            self.num_cities = len(self.G.nodes)
            for i in range(1, self.num_cities):
                for p in range(1, self.num_cities):
                    self.qb.binary_var(f"x_{i}_{p}")
            self.var_names = self.qb.variables


    def _define_objective_from_maxcut_graph(self):
        """Constrói MaxCut: maximize sum_w [ x_u + x_v - 2 x_u x_v ],
        tratando arestas com 'aux' como termos lineares quando o aux é fixo."""
        offset = 0
        linear = {}
        quadratic = {}

        for u, v, data in self.G.edges(data=True):
            w = data.get("weight", 0.0)
            if w == 0:
                continue

            # Arestas envolvendo 'aux': viram termos lineares se aux está fixo
            if self.variable_aux_fixed and ("aux" in (u, v)):
                other = v if u == "aux" else u
                # -w * x_other  (constante +w ignorada)
                offset += w
                linear[other] = linear.get(other, 0.0) - float(w)
                continue

            # Arestas "reais" (sem aux) -> forma padrão do MaxCut
            linear[u] = linear.get(u, 0.0) + float(w)
            linear[v] = linear.get(v, 0.0) + float(w)

            # termo quadrático -2 w x_u x_v
            key = (u, v) if u <= v else (v, u)
            quadratic[key] = quadratic.get(key, 0.0) - 2.0*float(w)

        # Importante: os nomes usados aqui DEVEM existir no QP
        self.qb.maximize(linear=linear, quadratic=quadratic, constant=offset)

    def _define_objective_function(self):
        linear_terms = {}
        quadratic_terms = {}

        if self.Maxcut:
            # Para cada aresta (i, j) com peso w:
            for i, j, w in self.G.edges(data="weight"):
                i_idx = self.variable_to_index[i]
                j_idx = self.variable_to_index[j]
                # soma w em x{i}, x{j}
                linear_terms[f"x{i_idx}"] = linear_terms.get(f"x{i_idx}", 0) + w
                linear_terms[f"x{j_idx}"] = linear_terms.get(f"x{j_idx}", 0) + w

                # termo quadratico -2 w * x_i x_j
                key = (f"x{i_idx}", f"x{j_idx}")
                quadratic_terms[key] = quadratic_terms.get(key, 0) - 2 * w

            self.qb.maximize(linear=linear_terms, quadratic=quadratic_terms)

        elif self.TSP:
            self.num_cities = self.n
            for i in range(1, self.num_cities):
                for j in range(1, self.num_cities):
                    if i != j:
                        for p in range(1, self.num_cities - 1):
                            quadratic_terms[(f"x_{i}_{p}", f"x_{j}_{p+1}")] = self.G[i][j]["weight"]
            # Contribution of city 0
            for i in range(1, self.num_cities):
                linear_terms[f"x_{i}_1"] = self.G[0][i]["weight"]
                linear_terms[f"x_{i}_{self.num_cities - 1}"] = self.G[0][i]["weight"]

            self.qb.minimize(linear=linear_terms, quadratic=quadratic_terms)


    def _add_constraints(self):
        if self.TSP:
            # Each city visited once
            for i in range(1, self.n):
                coeffs = {f"x_{i}_{p}": 1 for p in range(1, self.n)}
                self.qb.linear_constraint(
                    linear=coeffs,
                    sense="==",
                    rhs=1,
                    name=f"constraint_city_{i}"
                )
            
            # Each position occupied by exactly one city
            for p in range(1, self.n):
                coeffs = {f"x_{i}_{p}": 1 for i in range(1, self.n)}
                self.qb.linear_constraint(
                    linear=coeffs,
                    sense="==",
                    rhs=1,
                    name=f"constraint_pos_{p}"
                )

            print(self.qb.prettyprint())


    def multibasis_encoding_multiplets_qiskit(self, m, k):
        """
        Determina quantos qubits (n_qubits) e quais Pauli (pauli_list) 
        para codificar 'm' variáveis binárias.
        """
        # pauli_ops = ["Z", "X", "Y"]
        n = k

        # Encontrar número mínimo de qubits necessário
        while math.comb(n, k) * len(self.pauli_ops) < m:
            n += 1
        
        pauli_list = []
        for qubits in itertools.combinations(range(n), k):
            for p_op in self.pauli_ops:  # Agora garantimos que todos os operadores são aplicados corretamente
                op_list = ["I"] * n
                for qb in qubits:
                    op_list[qb] = p_op
                pauli_str = "".join(op_list)
                pauli = Pauli(pauli_str)
                pauli_list.append(pauli)
                if len(pauli_list) >= m:
                    return pauli_list, n  # Retorna assim que o limite for atingido

        return pauli_list, n


    def convert_qp_to_qubo(self, penalty=1e5):
        converter = QuadraticProgramToQubo(penalty=penalty)
        qubo = converter.convert(self.qb)
        print(qubo.prettyprint())
        return qubo

    def create_maxcut_graph(self, qubo):
        G_maxcut = nx.Graph()
        variables = [v.name for v in qubo.variables]

        for var in variables:
            G_maxcut.add_node(var)
        aux_node = "aux"
        G_maxcut.add_node(aux_node)
        return G_maxcut, aux_node, variables

    def add_edges_maxcut(self, qubo, G_maxcut, aux_node, variables):
        quadratic_dict = qubo.objective.quadratic.to_dict()
        linear_dict = qubo.objective.linear.to_dict()
    
        for (i_idx, j_idx), q_val in quadratic_dict.items():
            if q_val == 0:
                continue
            var1 = variables[i_idx]
            var2 = variables[j_idx]

    
            if var1 != var2:
                if G_maxcut.has_edge(var1, var2):
                    G_maxcut[var1][var2]["weight"] += q_val/4
                else:
                    G_maxcut.add_edge(var1, var2, weight=q_val/4)
            else:
                # diagonal
                # print('Diagonal:', i_idx)
                
                if G_maxcut.has_edge(aux_node, var1):
                    G_maxcut[aux_node][var1]["weight"] += q_val/2
                else:
                    G_maxcut.add_edge(aux_node, var1, weight= q_val/2)
    
        for i_idx, c_val in linear_dict.items():
            if c_val == 0:
                continue
            var_name = variables[i_idx]
    
            # Inicializa o peso com o termo linear
            weight = c_val/2  
    
            # Soma todas as conexões que a variável possui no QUBO
            for j_idx, q_val in quadratic_dict.items():
                if (j_idx[0] == i_idx or j_idx[1] == i_idx) and (j_idx[0] != j_idx[1]):  # Se a variável i interage com j
                    weight += q_val/4
                # if j_idx[0] == j_idx[1] == i_idx:
                #     print('Diagonal:', j_idx[0])
    
            # Adiciona ou atualiza a aresta do nó auxiliar
            if G_maxcut.has_edge(aux_node, var_name):
                G_maxcut[aux_node][var_name]["weight"] += weight
            else:
                G_maxcut.add_edge(aux_node, var_name, weight= weight)
    
        return G_maxcut 

    def convert_tsp_to_maxcut(self):
        """
        Converte TSP -> QUBO -> Grafo MaxCut, retorna (G_maxcut, variables).
        """
        qubo = self.convert_qp_to_qubo()
        print("QUBO:\n", qubo.prettyprint())

        G_maxcut, aux_node, variables = self.create_maxcut_graph(qubo)
        G_maxcut = self.add_edges_maxcut(qubo, G_maxcut, aux_node, variables)
        return G_maxcut, variables


    def maxcut_solution_to_tsp(self, bitstring, variables):
        """
        Interpreta bitstring (0/1) indexada pela 'variables' no formato 'x_{i}_{p}'.
        Ignora 'aux'. Devolve [rota].
        """
        if "aux" in variables:
            variables = [v for v in variables if v != "aux"]

        var_to_index = {v: i for i, v in enumerate(variables)}
        qubit_index = {}  # Ensure qubit_index is defined

        for var, index in var_to_index.items():
            _, i, p = var.split('_')  # Extract i and p from the key
            i, p = int(i), int(p)  # Convert to integers
            qubit_index['x', i, p] = index  # Assign value 

        flipped_bitstring = bitstring
        
        if not self.variable_aux_fixed:
            if bitstring[-1] == '0':
                flipped_bitstring = ''.join('1' if b == '0' else '0' for b in bitstring)
            else:
                flipped_bitstring = bitstring



        route = interpret_solution_tsp(qubit_index, flipped_bitstring, self.n)           


        if route is not None:
            cost = sum(self.adj_matrix[route[i], route[(i + 1) % self.n]] for i in range(self.n))
        else:
            cost = None

        return route, cost, flipped_bitstring

    def configure_backend(self):
        if self.config.SIMULATION == "True":
            if not self.fake_backend:
                print("Proceeding with simulation...")
                        # Acrescenta erro de phase damping
                backend = AerSimulator(method="statevector") 
                # backend = AerSimulator(method="statevector", device='GPU')                            
            else:
                print("Proceeding with simulation in Fake IBM_Brisbane using AerSimulator...")

                # Local e estático -> Não gasta o limite da conta gratuita no IBM Quantum Cloud
                fake_backend = FakeBrisbane()    
                backend = AerSimulator.from_backend(fake_backend)
                # backend = AerSimulator.from_backend(fake_backend, device='GPU')

                # Online e dinâmico
                # service = QiskitRuntimeService(
                #     channel="ibm_quantum_platform", token=self.config.QXToken
                # )                
                # real_backend = service.backend("ibm_brisbane")
                
                # backend = AerSimulator.from_backend(real_backend, device='GPU')
                # backend = AerSimulator.from_backend(real_backend)

            # backend = QasmSimulator()
            backend.set_options(seed_simulator=self.seed)
        else:
            print("Proceeding with IBM Quantum hardware...")
            service = QiskitRuntimeService(
                channel="ibm_quantum_platform", token=self.config.QXToken
            )
            backend = service.least_busy(min_num_qubits=127, operational=True, simulator=False)
            # backend = service.backend("ibm_brisbane")
            print(f"Connected to {backend.name}!")
        return backend


    def configure_sampler(self, backend):
        """Decide entre SamplerV2 ou Runtime Sampler com sessão"""
        
        if self.config.SIMULATION == "False":
            # Estamos em hardware real → usar Runtime Sampler com session
            print("Configurando Runtime Sampler com sessão IBM...")
            service = QiskitRuntimeService(channel="ibm_quantum", token=self.config.QXToken)
            session = Session(service=service, backend=backend)
            sampler = RuntimeSampler(session=session)
            self._runtime_session = session  # Armazena para encerrar depois
        else:
            # Simulação (com ou sem ruído) → usar BackendSampler normalmente - infelizmente não funciona com o SamplerV2 do qiskit-ibm-runtime! 
            print("Configurando SamplerV2 local...")
            sampler = BackendSampler(backend=backend)
            sampler.options.default_shots = 1024

        return sampler

    def cost_func_estimator_expvals(self, params):

        qc = self.vqe_circuit

        # transform the observable defined on virtual qubits to
        # an observable defined on all physical qubits
        observable_isa = []
        for pauli_string in self.pauli_strings:


            observable_isa.append(pauli_string.apply_layout(layout=qc.layout))        

        job = self.estimator.run([(qc, observable_isa, [params])])
        pub_result = job.result()[0]
        expvals = pub_result.data.evs

        return expvals
      

    def regularization_term(self):
        m = len(self.expvals)  # Total number of vertices
        tanh_squared = [np.tanh(self.alpha * exp_val)**2 for exp_val in self.expvals]
        normalized_sum = np.sum(tanh_squared) / m
        return self.beta * self.nu * (normalized_sum**2)

    
    def relax_problem(self, problem) -> QuadraticProgram:
        """Change all variables to continuous."""
        relaxed_problem = copy.deepcopy(problem)
        for variable in relaxed_problem.variables:
            variable.vartype = VarType.CONTINUOUS

        return relaxed_problem 

    def regularization_parameter(self, c_star):
        if self.epsilon <= float(c_star) <= 1 - self.epsilon:
            c_star_regularized = int(c_star)
        elif float(c_star) < self.epsilon:
            c_star_regularized = self.epsilon
        else:  # c_star > 1 - self.epsilon
            c_star_regularized = 1 - self.epsilon
        return c_star_regularized 


    def loss_function_multibasis_maxcut(self, G, params, p, regularization_term=True):

        self.expvals = self.cost_func_estimator_expvals(params)

        # self.brick_layer_ansatz_vqe(num_qubits, reps)

        loss = 0
        # Compute the weighted sum of tanh terms
        for i, j, w in G.edges(data='weight'):
            # Se a aresta envolve o nó auxiliar, não a ignore; incorpore sua contribuição.
            if i == 'aux' and j != 'aux' and self.variable_aux_fixed:
                j_idx = self.variable_to_index[j]  
                tanh_j = np.tanh(self.alpha * self.expvals[j_idx])
                tanh_i = 1 # O valor esperado da variável fixa será sempre 1
                weight_factor = 1
            elif j == 'aux' and i != 'aux' and self.variable_aux_fixed:
                i_idx = self.variable_to_index[i]
                tanh_i = np.tanh(self.alpha * self.expvals[i_idx])
                tanh_j = 1 # O valor esperado da variável fixa será sempre 1
                weight_factor = 1
            # Arestas entre nós "reais":
            else:          
                i_idx = self.variable_to_index[i]  # Converte 'x_i_p' para um índice numérico
                j_idx = self.variable_to_index[j]
                tanh_i = np.tanh(self.alpha * self.expvals[i_idx])
                tanh_j = np.tanh(self.alpha * self.expvals[j_idx])                

                if self.warm_starting:
                    # print(f'self.c_stars_i{i}, self.c_stars_j{j}')
                    weight_factor = 1 + np.abs(self.c_stars[i_idx] - self.c_stars[j_idx]) 
                    # weight_factor = 1 + lambda_bias * (self.c_stars[i_idx] - self.c_stars[j_idx])**2
                else:
                    weight_factor = 1

            loss += weight_factor * w * tanh_i * tanh_j

        if regularization_term:
            loss += self.regularization_term()         

        return loss 


    def time_execution_feasibility(self, backend):
        # Retrieve backend properties
        properties = backend.properties()

        # Extract gate durations
        gate_durations = {}
        for gate in properties.gates:
            gate_name = gate.gate
            if gate.parameters:
                duration = gate.parameters[0].value  # Duration in seconds
                gate_durations[gate_name] = duration

        print("Gate durations (in seconds):")
        for gate, duration in gate_durations.items():
            print(f"{gate}: {duration * 1e9:.2f} ns")

        # Calculate total execution time
        total_time = 0
        for instruction, qargs, cargs in self.vqe:
            gate_name = instruction.name
            gate_time = gate_durations.get(gate_name, 0)
            total_time += gate_time

        print(f"Total circuit execution time: {total_time * 1e6:.2f} µs")

        # Extract coherence times with qubit indices
        coherence_times = {}
        for qubit_index, qubit in enumerate(properties.qubits):
            T1 = None
            T2 = None
            for param in qubit:
                if param.name == 'T1':
                    T1 = param.value
                elif param.name == 'T2':
                    T2 = param.value
            coherence_times[qubit_index] = {'T1': T1, 'T2': T2}
            print(f"Qubit {qubit_index}: T1 = {T1*1e6:.2f} µs, T2 = {T2*1e6:.2f} µs")

        # Access the layout to map virtual qubits to physical qubits
        transpile_layout = self.vqe._layout  # Note the underscore before 'layout'

        layout = transpile_layout.final_layout
        
        # Retrieve the virtual-to-physical qubit mapping
        virtual_to_physical = layout.get_virtual_bits()

        # Determine which physical qubits are used in the circuit
        used_physical_qubits = set(virtual_to_physical.values())

        # Now, get the minimum T1 and T2 among the used physical qubits
        min_T1 = min(coherence_times[q_index]['T1'] for q_index in used_physical_qubits)
        min_T2 = min(coherence_times[q_index]['T2'] for q_index in used_physical_qubits)

        # Compare execution time to thresholds
        threshold_T1 = 0.1 * min_T1
        threshold_T2 = 0.1 * min_T2

        print(f"Thresholds: 10% T1 = {threshold_T1*1e6:.2f} µs, 10% T2 = {threshold_T2*1e6:.2f} µs")
        print(f"Circuit execution time: {total_time*1e6:.2f} µs")

        if total_time < threshold_T1 and total_time < threshold_T2:
            print("Execution time is within acceptable limits.")
        else:
            print("Execution time may be too long; consider optimizing your circuit.")    
   
    def MS_gate(self, theta, qubit_i, qubit_j):
        """
        Implementa a porta Mølmer-Sørensen (MS) parcialmente emaranhadora no Qiskit.
    
        Parâmetros:
            theta (float): Parâmetro de emaranhamento.
            qubit_i (int): Índice do primeiro qubit.
            qubit_j (int): Índice do segundo qubit.
    
        Retorna:
            Gate: Porta personalizada representando a interação MS.
        """
        # Criar um circuito com dois qubits (necessário apenas para a porta MS)
        qc = QuantumCircuit(2)
    
        # Mudar para a base XY com rotação Y
        qc.ry(np.pi / 2, 0)
        qc.ry(np.pi / 2, 1)
    
        # Aplicar interação IsingXX (equivalente à MS gate)
        qc.rxx(2 * theta, 0, 1)
    
        # Reverter a base XY
        qc.ry(-np.pi / 2, 0)
        qc.ry(-np.pi / 2, 1)
    
        # Retornar como uma porta personalizada
        return qc.to_gate(label="MS_gate")
    
    def brick_layer_ansatz_vqe(self, p, theta = None):
        self.num_parameters = 0
        
        if theta is None:
            if self.n_qubits % 2 == 0:
                if p % 2 == 0:
                    self.num_parameters = p * self.n_qubits + (p/2) * (self.n_qubits - 1)
                else:
                    self.num_parameters = p * self.n_qubits + ((p-1)/2) * (self.n_qubits - 1) + self.n_qubits/2
            else:
                self.num_parameters = p * (self.n_qubits + (self.n_qubits - 1)/2)
            # for layer in range(p):
            #     for qubit in range(self.n_qubits - 1):
            #         if (layer + qubit) % 2 == 0:  # Alternar entre camadas
            #             num_parameters += 1
            theta = ParameterVector("θ", int(self.num_parameters))
        
        qc = QuantumCircuit(self.n_qubits)
        param_index = 0

        for layer in range(p):
        # 1) single‑qubit rotation layer – axis cycles X→Y→Z
            axis = layer % 3
            for qubit in range(self.n_qubits):
                if axis == 0:
                    qc.rx(theta[param_index], qubit)
                elif axis == 1:    
                    qc.ry(theta[param_index], qubit)
                else:    
                    qc.rz(theta[param_index], qubit)
                param_index += 1                
    
            # Camada de portas MS (estrutura de tijolos)
            for qubit in range(self.n_qubits - 1):
                if (layer + qubit) % 2 == 0:  # Alternar entre camadas
                    ms_gate = self.MS_gate(theta[param_index], qubit, qubit + 1)
                    qc.append(ms_gate, [qubit, qubit + 1])
                    param_index += 1
    
        return qc      

    def solve_problem(self, p=1, brickwork_circuit=False):

        backend = self.configure_backend()

        self.estimator = Estimator(backend)
        self.estimator.options.default_shots = 1024    

        if self.config.SIMULATION == False:
            # Set simple error suppression/mitigation options
            self.estimator.options.dynamical_decoupling.enable = True
            self.estimator.options.dynamical_decoupling.sequence_type = "XY4"
            self.estimator.options.twirling.enable_gates = True
            self.estimator.options.twirling.num_randomizations = "auto"                  

        # Define a callback function to track progress
        def callback(params):
            print(f"Current parameters: {params}")

        # Set up QAOA with the callback
        np.random.seed(self.seed)

        # Define the seed that will be used in the optimization process
        algorithm_globals.random_seed = self.seed 

        bitstring_GW, self.cut_value_GW = '', 1                    


        # Aqui temos um problema: Quando se fixa o aux e “colapsa” (aux, i) em offset + termo linear, ficamos com 16 variáveis 
        # e linear não pareado. Isso não é um MaxCut, pois é um qp com lineares “soltos”. 
        # Como GW foi projetado para um qp que seja Maxcut, o resultado acaba não sendo muito bom.
         
        if self.warm_starting and not self.TSP:
            if self.c_stars is None:
                optimizer = GoemansWilliamsonOptimizer(
                    num_cuts= self.num_cuts, 
                    sort_cuts=True,   # ordena do melhor para o pior
                    unique_cuts=False, # remove duplicados                    
                    seed=42)
                result = optimizer.solve(self.qb)
                bitstring_GW = ''.join(str(int(float(bit))) for bit in result.x)
                self.cut_value_GW = result.fval        

                mean_val = 0     
                all_cuts = [(s.fval, s.x) for s in result.samples]       
                for fval, x in all_cuts:
                    # print(fval, x)
                    mean_val += fval     
                self.mean_GW =  mean_val/self.num_cuts 
                              
                c_stars_pure = [self.regularization_parameter(c_star) for c_star in bitstring_GW]        
                if c_stars_pure[-1] == 0 and self.variable_aux_fixed != True:   
                    self.c_stars = [1-c_star for c_star in c_stars_pure]
                else:
                    self.c_stars = c_stars_pure


            print(self.c_stars)       

        # Reseting the seed
        np.random.seed(self.seed)


        if brickwork_circuit == False:
            self.num_parameters = 4*self.n_qubits
            if p>1:
                self.num_parameters += (p-1)*self.n_qubits*2
            vqe = TwoLocal(self.n_qubits, ['ry', 'rz'], 'cx', 'linear', reps=p, insert_barriers=True)
            initial_params = np.random.uniform(0, 2 * np.pi, int(self.num_parameters)) #Menos da Metade dos parâmetros do original
        else:
            vqe = self.brick_layer_ansatz_vqe(p)
            initial_params = np.random.uniform(0, 2 * np.pi, int(self.num_parameters))     

        # with open(f'parametros_{self.epsilon}.txt', mode='+a', encoding='utf8') as f:
        #     f.write(str(initial_params) + '\n')

        # vqe.draw('mpl').savefig("init_qc.png")           

        # Create a custom pass manager
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=42)
        # pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

        # Transpile the circuit
        self.vqe_circuit = pm.run(vqe)    

        two_qubit_gates = 0
        for gate in self.vqe_circuit.data:
            if len(gate[1]) == 2:  # Verifica se a porta envolve dois qubits
                two_qubit_gates += 1

        print(f"Número de portas de 2 qubits: {two_qubit_gates}")           

        # self.vqe_circuit.draw('mpl').savefig("init_qc_optimized.png")   

        print('Circuit Depth:', self.vqe_circuit.depth())

        print('Núm. Qubits:', self.vqe_circuit.num_qubits)



        if self.fake_backend or self.config.SIMULATION == False:
            try:
                self.time_execution_feasibility(backend)
            except:
                pass                         
                                          
        def objective(params):
            if self.Maxcut:
                cost = self.loss_function_multibasis_maxcut(self.G, params, p)
            elif self.TSP:
                cost = self.loss_function_multibasis_tsp(params, p)              

            return cost

        # Perform optimization using COBYLA
        result = minimize(
            objective,
            initial_params,
            # args= (pauli_expectations, weights, edges, alpha)
            method='COBYLA',
            options={
                'maxiter': 1000,
                'disp': True,
                'rhobeg': 0.1,  # Initial step size
                'tol': 1e-6
            },
            callback=callback
        )                              
        
        # Considerando o resultado ótimo da função de custo encontrado pelo otimizador, usamos os parâmetros ótimos encontrados (result.x)
        # para calcular o valor esperado dos observáveis do circuito quântico (PCE). Depois consideramos os sinais desses valores para 
        # definir se nosso bit será 0 (sinal -) ou 1(sinal +). ---------------------------------------------------------------------------                
        
        self.expvals = self.cost_func_estimator_expvals(result.x)

        bitstring = ""
        for val in self.expvals:
            # If sign is negative => bit = 0, else bit = 1 (or whichever convention you want)
            bit = 1 if val >= 0 else 0
            bitstring += str(bit)
        # -------------------------------------------------------------------------------------------------------------------------------


        # bitstring = bitstring[0:-1]

        flipped_bitstring = bitstring
        
        if not self.TSP:
            if not self.variable_aux_fixed:
                if bitstring[-1] == '0':
                    flipped_bitstring = ''.join('1' if b == '0' else '0' for b in bitstring)
                else:
                    flipped_bitstring = bitstring        

        print('Proceeding with the bit swap search...')
        new_bitstring, new_cost, old_cost = self.bit_swap_search(flipped_bitstring)


        print(f'Current bitstring:{bitstring}')
        print(f'Improved bitstring:{new_bitstring}')

        if not self.TSP:
            # no caso do cut_value_GW, provavelmente será negativo no caso de fixarmos o nó auxiliar
            return flipped_bitstring, old_cost, new_bitstring, new_cost, bitstring_GW, self.cut_value_GW, self.c_stars
        else:
            return flipped_bitstring, old_cost, new_bitstring, new_cost



    def bit_swap_search(self, current_bitstring):
        # Initialize cost for the current solution
        if not hasattr(self, "qubit_index"):
            self.qubit_index = {}
            idx = 0
            for i in range(1, self.n):
                for p in range(1, self.n):
                    self.qubit_index[('x', i, p)] = idx
                    idx += 1            
        if self.Maxcut:
            current_cost_value, _, _ = interpret_maxcut_solution(current_bitstring, self.G, self.variable_to_index)
        else:
            tour = interpret_solution_tsp(self.qubit_index, current_bitstring, self.n)
            if tour is not None:
                current_cost_value = sum(self.adj_matrix[tour[i], tour[(i + 1) % self.n]] for i in range(self.n))
            else:
                current_cost_value = 0

        best_cost_value = current_cost_value

        improved_bitstring = current_bitstring[:]
        improved_cost = best_cost_value        
            
        
        # Perform single-bit swap search
        for i in range(len(current_bitstring)):
            # Flip the bit
            new_bitstring_list = list(current_bitstring)
            bit_list = list(current_bitstring)
            if bit_list[i] == '0':
                new_bitstring_list[i] = '1'
            else:
                new_bitstring_list[i] = '0'  

            new_bitstring = ''.join(new_bitstring_list)

            # Calculate the new cost
            if self.Maxcut:
                new_cost, new_cut_edges, new_partition = interpret_maxcut_solution(new_bitstring, self.G, self.variable_to_index)
            else:
                tour = interpret_solution_tsp(self.qubit_index, new_bitstring, self.n)
                if tour is not None:
                    new_cost = sum(self.adj_matrix[tour[i], tour[(i + 1) % self.n]] for i in range(self.n))
                else:
                    new_cost = 0


            # Funciona para TSP e Maxcut avaliar se o custo novo é maior que o velho, 
            # pois, se o custo velho é válido, não há como flipando um único bit, conseguir um resultado melhor que também seja válido!
            # Como todo resultado inválido é setado como 0, qualquer resultado acima disso é um avanço.
            # Assim, se o custo novo for maior que o antigo, significa que o antigo é 0.
            if new_cost > best_cost_value:
                if self.Maxcut:
                    improved_cost, improve_cut_edges, improved_partition = new_cost, new_cut_edges, new_partition
                else:
                    improved_cost = new_cost
                improved_bitstring = new_bitstring
                best_cost_value = new_cost

        print('Current cost: ', current_cost_value)
        print('Improved cost: ', improved_cost)

        return improved_bitstring, improved_cost, current_cost_value


# #######################################################################################
# A função abaixo é incompatível com modelos de ruído no qiskit.
# O SamplingVQE acaba gerando matrizes não-hermitianas!
# Mas funciona com Fakebackend.
# #######################################################################################

    def solve_problem_standard(self, p=1):

        converter = QuadraticProgramToQubo()
        qubo = converter.convert(self.qb)

        print('Number of qubits: ', len(qubo.variables))

        # Print the Ising model
        print(qubo.to_ising())

        qubo_ops, offset = qubo.to_ising()        

        backend = self.configure_backend()       
       

        sampler = self.configure_sampler(backend)

        if self.config.SIMULATION == False:
            # Set simple error suppression/mitigation options
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = "auto"          

        # Define a callback function to track progress
        def callback_vqe(eval_count, params, value, meta):
            print(f"Iteration: {eval_count}")
            print(f"Current parameters: {params}")
            print(f"Current energy: {value}")
            print(f"Meta info: {meta}")

        # Set up QAOA with the callback
        np.random.seed(self.seed)

        # Define the seed that will be used in the optimization process
        algorithm_globals.random_seed = self.seed        

        # Generate initial parameters using the seed         
        ansatz = TwoLocal(
            len(self.qb.variables),
            rotation_blocks=['ry', 'rz'],
            entanglement_blocks="cx",
            entanglement="linear",
            reps=p,
        )
        initial_parameters = np.random.random(ansatz.num_parameters)  

        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=self.seed)

        circuit = pm.run(ansatz)

        # if self.fake_backend:
        #     circuit.draw('mpl').savefig('init_qc_optimized_vqe_fakebrisbane')
        # else:
        #     circuit.draw('mpl').savefig('init_qc_optimized_vqe')     

        print('Circuit Depth:', circuit.depth())
        print('Núm. Qubits:', circuit.num_qubits)

        two_qubit_gates = 0
        for gate in circuit.data:
            if len(gate[1]) == 2:  # Verifica se a porta envolve dois qubits
                two_qubit_gates += 1

        print(f"Número de portas de 2 qubits: {two_qubit_gates}")            

        print(ansatz.num_parameters)

        optimizer = COBYLA()

        sampling_vqe = SamplingVQE(
            sampler=sampler,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_parameters,
            callback=callback_vqe,  # if you have a callback function defined
        )

        start_time = time.time()
        result = sampling_vqe.compute_minimum_eigenvalue(operator=qubo_ops)
        end_time = time.time()

        self.execution_time = end_time - start_time
        print(f"VQE ground state energy (no offset): {result.eigenvalue.real:.5f}")
        print(
            f"VQE ground state energy (with offset): {(result.eigenvalue + offset).real:.5f}"
        )
        print(f"VQE bitstring solution: {result.eigenstate}")

        print(f"Execution time: {self.execution_time:.4f} seconds")

        # Find the integer key with the largest probability:
        most_likely_int = max(result.eigenstate, key=result.eigenstate.get)  # e.g. 15
        max_probability = result.eigenstate[most_likely_int]          # e.g. 0.783203125

        # Convert that integer to a bitstring of length 4
        # By default, '{:b}' omits leading zeros, so we use zfill to enforce num_qubits bits:
        raw_bitstring = f"{most_likely_int:0{self.n}b}"  # "1111"

        bitstring = raw_bitstring.zfill(len(qubo.variables))

        # Qiskit geralmente numera os bits da direita para a esquerda,
        # mas nesse caso isso não importa, pois a rota será interpretada de acordo com
        # o dicionário self.qubit_index  

        # Retrieve the optimal parameters
        optimal_params = result.optimal_point
        print("Optimal parameters:", optimal_params)

        flipped_bitstring = bitstring
        
        # Tanto faz fixa 0 ou 1, porque em tese, como o Maxcut é simétrico, ambas as escolhas 
        # retornarão resultados estatísticamente semelhantes.
        if not self.TSP:
            if not self.variable_aux_fixed:
                if bitstring[-1] == '1':
                    flipped_bitstring = ''.join('1' if b == '0' else '0' for b in bitstring)
                else:
                    flipped_bitstring = bitstring  

        print('Proceeding with the bit swap search...')
        new_bitstring, new_cost, old_cost = self.bit_swap_search(flipped_bitstring)

        # old_cost = result.fun

        route_result = interpret_solution_tsp(self.qubit_index, new_bitstring, self.n)      

        return flipped_bitstring, old_cost, new_bitstring, new_cost, route_result










    
