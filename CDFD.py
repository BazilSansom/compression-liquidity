import numpy as np
import networkx as nx

# needed for BFF
from scipy.sparse import csr_array, coo_array, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import norm
from discreteMarkovChain import markovChain

# Needed for pulp
import pulp as pl
from scipy.sparse import lil_array, csr_array # coo_array,

# Needed for ortools
from ortools.graph.python import min_cost_flow
#from scipy.sparse import coo_array, csr_array


# Main fucntions

def get_circularity (C, D):  
    """Gives circularity ratio of decompositions (C,D). 
    
    Parameters
    ----------
    C : nx.DiGraph, sparse matrix or array like
        The circular part.
    D : nx.DiGraph, sparse matrix or array like
        The directional part.  
    
    Returns
    -------
    circularity_ratio : float
        The circularity ratio. 
    """
    C = _convert_to_csr_array(C)
    D = _convert_to_csr_array(D)
    C_weight = np.sum(C.data)
    D_weight = np.sum(D.data)
    circularity_ratio = C_weight/(C_weight+D_weight)
    return circularity_ratio


def get_directedness (C, D):  
    """Gives directedness ratio of decompositions (C,D). 
    
    Parameters
    ----------
    C : nx.DiGraph, sparse matrix or array like
        The circular part.
    D : nx.DiGraph, sparse matrix or array like
        The directional part.  
    
    Returns
    -------
    directedness_ratio : float
        The directedness ratio. 
    """
    directedness_ratio = 1 - get_circularity(C,D)
    return directedness_ratio

def is_decomposition(C, D, G = None, TOL = 1e-8 ): # TO DO: extend to accept nx.DiGraph and arrays
    """Checks if (C,D) is a valid decomposition. 
    
    Parameters
    ----------
    C : nx.DiGraph, sparse matrix or array like
        The circular part.
    D : nx.DiGraph, sparse matrix or array like
        The directional part.  
    G : nx.DiGraph or None (default = None)
        The graph that C and D are supposed to be a decomposition of. If None this condition is not checked. 
    TOL : float (default = 1e-8)
        The tolerance to accept that C is circular and C+D = W using frobenius norm of matrices. 
    
    Returns
    -------
    is_decomposition : bool
        True if it is a decomposition. 
    """ 
    C = _convert_to_csr_array(C)
    D = _convert_to_csr_array(D)
    if G is None:
        W = C + D
    else: 
        W = _convert_to_csr_array(G)
    
    checks_dic = _checks_decomposition_dic(C, D, W)
    is_acyclic = checks_dic['is_acyclic']
    is_circular = checks_dic['balance_error'] < TOL 
    sums_to_W = checks_dic['sums_to_W_error'] < TOL
    
    if not is_acyclic:
        print("Directional part is not acyclic.")
    if not is_circular:
        print(f"Circular part is not balanced by {checks_dic['balance_error']}. You may want to change TOL.")
    if not sums_to_W:
        print(f"The decomposition doesn't add up to G by {checks_dic['sums_to_W_error']}. You may want to change TOL.")
        
    is_decomp = is_acyclic and is_circular and sums_to_W
    return is_decomp

# ADD maximal as solution method and choose pulp!
def CDFD ( G, solution_method = "BFF", TOL_ZERO = 1e-12, TOL_decimals = 1e-8, MAX_decimals = 6 ): # TO DO: make relative TOL (use min(W.data) and count significant figures from there)
    """Gets CDFD decomposition of graph.

    Parameters
    ----------

    G : nx.DiGraph
        Graph we are interested in. Note that nx.Graph would be completely circular.
    solution_method : str ("BFF", "Maximal", "min_cost_pulp", "min_cost_ortools", "HA", "HAMD", "L2-closest-C")
        Method used to find a decomposition (affect which one we find).
        "BFF" : Balanced Flow-based (default, iterative)
        "Maximal" or "min_cost_pulp" : Maximum directional flow (min-cost flow with PuLP)
        "min_cost_ortools" : For integer-like data (faster for money with 2 decimal places)
        "HA" : Hodge-Aligned (L2-small, closest to HHD)
        "HAMD" : Hodge-Aligned Maximum Directional (L1-larger-D)
        "L2-closest-C" : L2-closest circular part (minimal circular)
    TOL_ZERO : float (degault = 1e-12)
        The tolerance to accept that a float is actually zero. As it is absolute, it should be modified, depending on the size of the weights in G.
        It is also used to define tolerance to accept solution.
    TOL_decimals : float (default = 1e-8)
        Only used when solution method is min_cost_ortools. Tolerance to accept that a float corresponds to an integer (after scaling).
    MAX_decimals : int (default = 6)
        Only used when in solution method is min_cost_ortools. Maximal number of decimal places we want to scale weights of W by.

    Returns
    -------
    C : csr_array
        Circulartion part of decomposition.
    D : csr_array
        Directional part of decomposition.
    """
    # Import hodge methods if needed
    if solution_method in ["HA", "HAMD", "L2-closest-C"]:
        from hodge import CDFD_L2_small, CDFD_HAMD, CDFD_L2_large

    sol_method_dic = {
        "BFF": CDFD_BFF,
        "min_cost_pulp": CDFD_min_cost_pulp,
        "min_cost_ortools": CDFD_min_cost_ortools,
        "Maximal": CDFD_min_cost_pulp
    }

    # Add Hodge methods if imported
    if solution_method in ["HA", "HAMD", "L2-closest-C"]:
        sol_method_dic["HA"] = CDFD_L2_small
        sol_method_dic["HAMD"] = CDFD_HAMD
        sol_method_dic["L2-closest-C"] = CDFD_L2_large

    assert solution_method in sol_method_dic, "Incorrect method specified. Choose from %r" % sol_method_dic.keys()

    W = _convert_to_csr_array ( G )

    # Hodge methods return (C, D, info), others return (C, D)
    if solution_method in ["HA", "HAMD", "L2-closest-C"]:
        result = sol_method_dic[solution_method](W, tol=TOL_ZERO)
        C, D = result[0], result[1]
    else:
        C, D = sol_method_dic[solution_method](W, TOL_ZERO = TOL_ZERO, TOL_decimals = TOL_decimals, MAX_decimals = MAX_decimals)
    
    type_Graphs = [type(nx.Graph()), type(nx.DiGraph())]
    if type(G) in type_Graphs:
        # For Hodge methods, don't copy graph attributes since edge structure may differ
        if solution_method in ["HA", "HAMD", "L2-closest-C"]:
            C = _convert_matrix_to_graph ( C )
            D = _convert_matrix_to_graph ( D )
        else:
            C = _convert_matrix_to_graph ( C , G )
            D = _convert_matrix_to_graph ( D , G )
    else:
        C = _convert_matrix_to_graph ( C )
        D = _convert_matrix_to_graph ( D )

    return C, D
    

#-----------------------------------------------

# For working with adjacency matrix directly

def CDFD_BFF(W, TOL_ZERO = 1e-12, **kwargs): 
    """Gets CDFD decomposition of an adjacency matrix of a graph by applying BFF recursively. 
    
    Parameters
    ----------
    W : array like or sparse matrix
        Weighted adjacency matrix of graph. 
    TOL_ZERO : float (default = 1e-12)
        The tolerance to accept that a float is actually zero. 
    **kwargs : 
        This arguments are ignored, but needed to avoid error in CDFD
    
    Returns
    -------
    C : csr_array
        Circulartion part of decomposition. 
    D : csr_array
        Directional part of decomposition. 
    """
    W = csr_matrix(W)
    
    # removing loops (will add them to circular part)
    loops = diags(W.diagonal())
    
    # separate isolated nodes
    W_no_isolated, W_isolated, idx_no_isolated = _separate_isolated(W - loops) 
    
    # Initialization
    Wtemp = W_no_isolated
    n_nodes = Wtemp.shape[0] # number of rows/colums
    n_edges = len(Wtemp.data)
    n_components = 0
    iterations = 0 
    Max_it = 2*n_edges + 1 # we should remove at least an edge in each iteration, factor 2 to have margin for rounding errors (+ 1 to avoid problems when n_edges = 0)
    
    # While there is still cycle structure in Wtemp 
    while n_components < n_nodes and iterations < Max_it: 
        # Find a circulation on Wtemp
        c, n_components = _BFF(Wtemp)
        # Remove circulation from what will become directional part
        Wtemp = Wtemp - c
        # Remove small values
        Wtemp.data[Wtemp.data < TOL_ZERO] = 0 
        Wtemp.eliminate_zeros()
        iterations += 1
    if iterations == Max_it :
        raise Exception("Maximum iterations was reached and decomposition was not found.")
    
    # restor isolated nodes to the matrix (all their edges are in directed part)
    Wtemp_reshaped = _sub_matrix(Wtemp, W, idx_no_isolated) 
    Wtemp_reshaped +=  W_isolated
    
    D = Wtemp_reshaped # TO DO: check the edges which values are very close to ones on W (so used all of them) and remove float errors by assigning exactly value in W. 
    D = csr_array(D)
    C = W - D # W already has loops 
    C.data[C.data < TOL_ZERO] = 0 
    C.eliminate_zeros()
    
    
    # Check is decomposition
    TOL_decomposition = 10 *2 *len(W.data) * TOL_ZERO   # needs to be this big to count for all rounding errors incurred (using L^2 norm <= L^1)
                                                        # Factor 2 as in balance one edge appears twice, factor 10 to have margin
    _check_decomposition_raise_Warning(C, D, W, TOL_decomposition )

    return C, D


def CDFD_min_cost_pulp ( W, TOL_ZERO = 1e-12, **kwargs ) : 
    """Gets decomposition with directional part being minimal-cost flow solution (all cost 1) using build in function in pulp. 
    Equivalently gets maximal compression. Suitable for real valued flows. 
    NOTE: Pulp uses c floats which have less precision, so we should expect at most 8 significant digits. TOL_ZERO should be then choosen considering this or a Warning will be raised. 
    
    Parameters
    ----------
    W : array_like or sparse matrix
        The adjacency matrix of the graph. 
    TOL_ZERO : float (default = 1e-12)
        Tolerance to accept that a float is actually 0. 
    **kwargs : 
        This arguments are ignored, but needed to avoid error in CDFD
    
    Returns
    -------
    C : csr_array
        Circulartion part of decomposition. 
    D : csr_array
        Directional part of decomposition. 
    """
    D = _min_cost_flow_pulp( W )
    # removing edges with weights sufficiently close to 0
    D.data[D.data < TOL_ZERO] = 0 
    D.eliminate_zeros()
    C = W - D 
    C.data[C.data < TOL_ZERO] = 0 
    C.eliminate_zeros()
    
    # Check is decomposition
    TOL_decomposition = 10 *2 *len(W.data) * TOL_ZERO   # needs to be this big to count for all rounding errors incurred (using L^2 norm <= L^1)
                                                        # Factor 2 as in balance one edge appears twice, factor 10 to have margin
                                                        # LLLL CHANGE BIG FACTOR 
    _check_decomposition_raise_Warning(C, D, W, TOL_decomposition )
    
    return C, D

def CDFD_min_cost_ortools (W, TOL_ZERO = 1e-12,  TOL_decimals = 1e-8, MAX_decimals = 6): 
    """Gets decomposition with directional part being minimal-cost flow solution (all cost 1) using build in function in ortools. 
    Equivalently gets maximal compression. Only suitable for flows close to integer values, such as money (with 2 decimal places). 
    
    Parameters
    ----------
    W : array_like or sparse matrix
        The adjacency matrix of the graph. Will be converted to coo_array. 
    TOL_ZERO : float (default = 1e-12)
        Tolerance to accept that a float is actually 0. Here it is only use to deduce a tolerance to accept decomposition as valid.  
    TOL_decimals : float (default = 1e-8)
        Tolerance to accept that a float corresponds to an integer (after scaling). 
    MAX_decimals : int (default = 6)
        Maximal number of decimal places we want to scale weights of W by. 
    
    Returns
    -------
    C : csr_array
        Circulartion part of decomposition. 
    D : csr_array
        Directional part of decomposition. 
    """
   
    D = _min_cost_flow_int( W )
    C = W - D 
    # No need to remove close to 0 as all done in int
    
    # Check is decomposition
    TOL_decomposition = 10 *2 *len(W.data) * TOL_ZERO   # needs to be this big to count for all rounding errors incurred (using L^2 norm <= L^1)
                                                        # Factor 2 as in balance one edge appears twice, factor 10 to have margin
    _check_decomposition_raise_Warning(C, D, W, TOL_decomposition )
    
    return C, D


#------------------------------------------------------------------------------------------------------------------------------------


# Internal functions 

# DO: DEFINE MY OWN NORM FUNCTION FROM DATA np.abs np.sum (check if numpy otherwise assum is sparse matrix).

# ----------------------------------------
# General

def _convert_to_csr_array ( G ): # TO DO: should add weight = 'weight' here and in all main functions (above)
    """Gives adjacency matrix of a graph or converts np.array adjacency matrix to csr_array format.  
    
    Parameters
    ----------
    G : nx.Graph, nx.DiGraph or array like
        The graph we want adjacency matrix of or adjecency matrix we want to change format of. 
        
    Returns
    -------
    W : csr_array
        Weighted adjacency matrix of graph.    
    """ 
    type_Graphs = [type(nx.Graph()), type(nx.DiGraph())]
    if type(G) in type_Graphs:
        W = csr_array(nx.adjacency_matrix (G))
    else:
        W = csr_array(G)
    
    return W

def _convert_matrix_to_graph ( W, graph_with_data = None , weight = 'weight'):
    """Given an adjacency matrix creates a graph with the weight given by it and the rest of the data taken from graph_with_data.
    
    Parameters
    ----------
    W : sparse matrix
        Weighted adjacency matrix of graph.
    graph_with_data : nx.DiGraph or None
        Graph we want to copy all data from.  Must contain all edges contained in W. If None a graph with no extra data is generated. 
    weight : string (default='weight')
        The edge data key used to save weights from W in G.
    
    Returns
    -------
    G : nx.DiGraph 
        The graph with adjacency matrix W. Note: the order of the edges is not preserved from graph_with_data.
    """ 
    if graph_with_data  is None:
        G = nx.DiGraph(W)
        return G
    G = nx.DiGraph()
    G.add_nodes_from (graph_with_data.nodes(data=True))
    node_mapping = list(G.nodes())
    W_coo = coo_array(W)
    W_edge_weight = {(node_mapping[i] , node_mapping[j], w) for i, j, w in zip(W_coo.row, W_coo.col, W_coo.data)}
    G_edges = []
    for a, b, w in W_edge_weight: 
        attrs = graph_with_data[a][b].copy() # only shallow (unlikely a deep one is needed)
        attrs[weight] = w
        G_edges.append((a,b, attrs)) 

    G.add_edges_from(G_edges)
    return G  

def _checks_decomposition_dic( C, D, W ): 
    """Computes values to check if  (C,D) is a valid decomposition. 
    
    Parameters
    ----------
    C : sparse matrix
        The matrix representing the circular part.
    D : sparse matrix
        The matrix representing directional part.  
    W : sparse matrix 
        The matrix that C and D are supposed to be a decomposition of. 
        
    Returns
    -------
    checks_dic : dic
        Dictionary with error terms (using frobenious norm) for C to be balanced, for C+D = W, and whether D is acyclic or not.   
    """ 
    checks_dic = {}
    # acyclic iff all nodes isolated (strongly connected)
    checks_dic['is_acyclic'] = D.shape[0] == connected_components(csgraph=D, directed=True, connection='strong')[0] and np.sum(D.diagonal()) == 0  # checking all isoleted strongly connected compoennts and no loops
    checks_dic['balance_error'] = np.linalg.norm(C.sum(axis=1).T - C.sum(axis=0))
    checks_dic['sums_to_W_error'] = norm(W - (C + D))
    return checks_dic
    
def _check_decomposition_raise_Warning ( C, D, W, TOL):
    """Checks if (C,D) for a valid decomposition an raises Exception otherwise. 
    
    Parameters
    ----------
    C : sparse matrix
        The matrix representing the circular part.
    D : sparse matrix
        The matrix representing directional part.  
    W : sparse matrix 
        The matrix that C and D are supposed to be a decomposition of. 
    TOL : float 
        The tolerance to accept that C is circular and C+D = W using frobenius norm of matrices. 
          
    """ 
    checks_dic = _checks_decomposition_dic(C, D, W)
    if not checks_dic['is_acyclic']: 
        raise Warning("Directional part is not acyclic.")
    if checks_dic['balance_error'] > TOL: 
        raise Warning(f"Circular part is not balanced by {checks_dic['balance_error']}. You may want to change TOL_ZERO.")
    if checks_dic['sums_to_W_error'] > TOL: 
        raise Warning(f"The decomposition doesn't add up to W by {checks_dic['sums_to_W_error']}. You may want to change TOL_ZERO.")

#-----------------------------------------------------------
# BFF 
# NOTE: we keep old format csr_matrix as both "connected_components" and "markovChain" work with this format
# due to this the code seems to fail with new format csr_array. Note that now code uses things like .multiply  
# and division matrix which may not work in new format

def _BFF(W):
    """Gets circulation from graph applying BFF to each strongly connected component. 
    This implementation uses the global form of the BBF algorithm. 
    
    Parameters
    ----------
    W : csr_matrix
        Weighted adjacency matrix of graph. 
    
    Returns
    -------
    C : csr_matrix
        Circulartion resulting from applying BFF once.
    n_components : int
        Number of stronlgy connected components. Used to detect if we need to apply _BBF again. 
    """
    W = W.copy()  # needed here, not sure why, but it is ok to add as doesn't effect performance. 
    # Get all strongly connected components in W with more than 1 node 
    n_components, labels = connected_components(csgraph=W, directed=True, connection='strong')  
    components_idx = _group_repeated_index_labels(labels)
    # initializin C data
    data =[] 
    row = []
    col = []
    # For each strongly connected component
    for idx in components_idx:  
        # Get the strongly connected component
        scc = W[idx, :][:, idx]
        # Get stationary distribution for this subgraph
        c_scc = _BFF_strongly_connected(scc)
        # Update C data
        _extend_matrix_data(c_scc, data, row, col, idx)
    #Construct C and convet to appropiate type    
    C = coo_array( (data, (row, col)), shape = W.shape)
    C = csr_matrix(C)
    return C, n_components

def _BFF_strongly_connected(W):
    """Gets circulation from a stronlgy connected graph using the BFF algorithm once.   
    
    Parameters
    ----------
    W : csr_matrix
        Weighted adjacency matrix of a strongly connected graph. 
    
    Returns
    -------
    c : csr_matrix
        Circulartion resulting from applying BFF once.  
    """
    row_sum_inv =  1/W.sum(axis=1) 
    P = csr_matrix(W.multiply(row_sum_inv)) # Normalizing adjacency matrix        
    stationary_distribution = _get_stationary_distribution(P)
    c_stationary = csr_matrix(P.multiply(stationary_distribution.T) ) # TO DO: could be more efficient keep format (as following function usese it)
                                                                      # for now keep always csr for consistemcy
    scaling_factors = _sparse_divide_nonzero(W, c_stationary)
    min_factor = np.min(scaling_factors.data) # NOTE: very important we do not feed 0 weight edges! otherwise this wont work
    c = min_factor * c_stationary 
    return c
    
    
def _get_stationary_distribution(P): 
    """Gets stationary distribution of a strongly connected markov chain.  
    
    Parameters
    ----------
    P : csr_matrix
        Transition matrix markov chain.
    
    Returns
    -------
    xFix : np.arrray [float]
        Stationary distribution.  
    """
    mc = markovChain(P)  
    mc.computePi('linear') #We can use 'linear', 'power', 'krylov' or 'eigen'. For us linear seems fastest. 
    pi = mc.pi
    xFix = pi.reshape(1,len(pi)) # converting to row vector
    return xFix

def _group_repeated_index_labels(array_labels):     
    """For each label that appears more than once in array_labels it creates and array with the indices where this label appears in array_labels. 
    
    Parameters
    ----------
    array_labels : np.array [int]
        Array with labels (some repeated).
    
    Returns
    -------
    grouped_idx : list [np.array [int]]
        list of array, each of them representing the indecies where a repeated label appeared in array_labels 
    """
    grouped_idx = _group_index_labels(array_labels)
    grouped_idx = filter(lambda x: x.size > 1, grouped_idx)
    return grouped_idx

def _group_index_labels(array_labels): 
    """For each label it creates and array with the indices where this label appears in array_labels. 
    
    Parameters
    ----------
    array_labels : np.array [int]
        Array with labels (some repeated).
    
    Returns
    -------
    grouped_idx : list [np.array [int]]
        list of array, each of them representing the indecies where a label appeared in array_labels 
    
    Example
    ---------
    >>> array_labels =  np.array([2,3,2])
    >>> _group_index_labels(array_labels)
    [np.array([0,2]), np.array([1])]
    """
    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(array_labels)
    # sorts records array so all unique elements are together using previous computation 
    sorted_records_array = array_labels[idx_sort]
    # returns the unique values, the index of the first occurrence of a value
    vals, idx_start = np.unique(sorted_records_array, return_index=True)
    # splits the indices into separate arrays
    grouped_idx = np.split(idx_sort, idx_start[1:])
    return grouped_idx

def _extend_matrix_data(M, data, row, col, idx): 
    """Saves the data in the matrix M in data, row, col, with the indices shifted by idx. 
    
    Parameters
    ----------
    M : sparse matrix 
        The matrix which information we want to save. Will be converted to coo_array.
    data : list
        List that we want to extend by adding the information in M.data
    row : list
        List that we want to extend by adding the indeces in M.row shifted by the correspondance in idx. 
    col : list
        List that we want to extend by adding the indeces in M.col shifted by the correspondance in idx. 
    """
    M_coo = coo_array(M)   
    shift_row = [idx[i] for i in M_coo.row] # Efficiency: significant time of BFF spend in the shifts
    shift_col = [idx[i] for i in M_coo.col]
    data.extend(M_coo.data)
    row.extend(shift_row)
    col.extend(shift_col)
    
def _separate_isolated(W):
    """Separetes isolated nodes in the sense of strongly connected component (that is they have their own strongly connected component). 
    
    Parameters
    ----------
    W : array like or sparse matrix 
        The adjacency matrix of the graph. Will be converted to csr_matrix.
   
    Returns
    -------
    W_no_isolated : csr_matrix
        Adjacency matrix of complete subgraph with the non-isolated nodes (as many rows as non-isolated nodes). 
    W_isolated : csr_matrix
        Adjacency matrix of all edges that have some isolated node, with shape W.shape. 
    indices_no_isolated: np.array [int]
        Indices of non-isolated nodes in W. 
    """
    W = W.copy()
    n_components, labels = connected_components(csgraph=W, directed=True, connection='strong') 
    components_idx = list(_group_repeated_index_labels(labels))
    if len (components_idx) == 0:
        indices_no_isolated = np.array([])
    else: 
        indices_no_isolated = np.concatenate(components_idx)
    
    W_no_isolated = W[indices_no_isolated, :][:, indices_no_isolated]
    W_no_isolated_reshape = _sub_matrix(W_no_isolated, W, indices_no_isolated)
    W_isolated = W-W_no_isolated_reshape # It seems that this both identifies 0 exactly (no float error) and eliminates zeros from data. 

    return W_no_isolated, W_isolated, indices_no_isolated 

# MM

def _sub_matrix(M_sub, M_big, idx): 
    """Puts matrix M_sub in a matrix of the size of M_big at idx indices (rest entries are implicitly 0). 
    Note that we don't use slicing as it isn't efficient for csr_matrix. 
    
    Parameters
    ----------
    M_sub :array like or sparse matrix
        The submatrix we want to expand. 
    M_big :array like or sparse matrix
        A matrix of the size of matrix we want to get .  
    idx : array like [int]
        The indices that give correspondences between row/col index in M_sub and M.   
    
    Returns
    -------
    M : csr_matrix
        Matrix with M_sub as submatrix.  
    
    Example
    ---------
    >>> M_sub = csr_matrix([[1,2],[3,4]]) 
    >>> M_big = csr_matrix((3,3))
    >>> idx = [0,2]
    >>> _sub_matrix(M_sub, M_big, idx).todense()
    matrix([[1., 0., 2.],
        [0., 0., 0.],
        [3., 0., 4.]])
    """
    M = coo_array(M_sub)
    shift_row = [idx[i] for i in M.row]
    shift_col = [idx[i] for i in M.col]
    M = coo_array( (M.data, (shift_row, shift_col)), shape = M_big.shape)
    return csr_matrix(M)
    
def _sparse_divide_nonzero(a, b):
    """Point wise division of a by b of the non-zero entries of sparse matrices. 
    
    Parameters
    ----------
    a : sparse matrix
        The matrix representing the numerator.
    b : sparse matrix
        The matrix representing the denominator. It should not have zeros in b.data, otherwise use b.eliminate_zeros().  
    
    Returns
    -------
    division : sparse matrix
               The quotient.
    """
    inv_b = b.copy()
    inv_b.data = 1 / inv_b.data
    division = a.multiply(inv_b)
    return division

#--------------------------------------------
# pulp

def _min_cost_flow_pulp (W):    
    """Gets minimal-cost flow solution using build in function in pulp. 
    Suitable for real valued flows. 
    
    Parameters
    ----------
    W : array_like or sparse matrix
        The adjacency matrix of the graph. 
    
    Returns
    -------
    D : csr_array
        Solution flow
    """
    
    # Create variables
    variables, variable_row_form, variable_column_form = _get_pulp_variables(W)
    
    # Set up problem in pl
    problem = pl.LpProblem("maximal_compression", pl.LpMinimize)
    # Objective
    problem += pl.lpSum(variables.values())
    # Constraints for net strength preservation
    out_strength = W.sum(axis=1)
    in_strength = W.sum(axis=0).T
    net_strength = out_strength - in_strength
    # Variable constrains for net preservation
    number_nodes = W.shape[0]  
    for i in range(number_nodes): 
        out_vars = pl.lpSum(variable_row_form[i])
        in_vars = pl.lpSum(variable_column_form[i])
        problem += (out_vars - in_vars == net_strength[i], f"NetStrength_{i}")
    solver = pl.PULP_CBC_CMD() # fracGap determines accuracy gapRel =0.001 # TO DO: Find best solver for minimum cost problem
    problem.solve(solver)
    
    # Check if found optimal
    if pl.LpStatus[problem.status] != 'Optimal': 
        raise Exception("Optimization failed.")
        
    # Store data in appropiate format 
    row = np.array([i for (i,j) in variables]) 
    col = np.array([j for (i,j) in variables])
    data = np.array([variable.varValue for variable in variables.values()])
    D = coo_array( (data, (row, col)), shape = W.shape)
    D = csr_array(D)
    
    return D

def _get_pulp_variables(W): 
    """Creates variables to be used in pl.LpProblem and stores in rows and columns so they can be added easily. 
     
    Parameters
    ----------
    W : array_like or sparse matrix
        The adjacency matrix of the graph. Will be converted to coo_array.
    
    Returns
    -------
    variables : dict
        Dictionary with all the variables indexed by tuples
    variable_row_form : list
        List of list of variables of each row
    variable_column_form : list
        List of list of variables of each column
    """
    # Convert to more covenient sparse format
    W_lil = lil_array(W) 
    W_lil_transpose  = W_lil.transpose()
    number_nodes = W.shape[0]  
    
    variable_row_form = []
    variables = {}
    for i, j_list, weight_list in zip(range(number_nodes),W_lil.rows,W_lil.data): 
        row_i = []
        for j, weight in zip(j_list,weight_list):
            variable_name = f"x_{i}_{j}"
            variable = pl.LpVariable(variable_name, lowBound=0, upBound=weight)
            row_i.append(variable)
            variables[(i, j)] = variable
        variable_row_form.append(row_i)

    variable_column_form = []
    for i, j_list in zip(range(number_nodes), W_lil_transpose.rows): 
        column_i = [variables[(j,i)] for j in j_list] # flip as work with transpose
        variable_column_form.append(column_i)
        
    return variables, variable_row_form, variable_column_form

#-------------------------------
#or-tools

# TO DO: should check if also fast for non integers by simply taking fix number of decimals. 
def _min_cost_flow_int (W, TOL_decimals = 1e-8, MAX_decimals = 6): 
    """Gets minimal-cost flow solution using build in function in ortools. 
    Only suitable for flows close to integer valued, such as money (with 2 decimal places). 
    
    Parameters
    ----------
    W : array_like or sparse matrix
        The adjacency matrix of the graph. Will be converted to coo_array. 
    TOL_decimals : float (default = 1e-8)
        Tolerance to accept that a float corresponds to an integer (after scaling). 
    MAX_decimals : int (default = 6)
        Maximal number of decimal places we want to scale weights of W by. 
    
    Returns
    -------
    D : csr_array
        Solution flow
    """
    
    W_coo = coo_array(W.copy())  
    
    # Convert to integer weights if needed 
    decimals = 0
    if np.issubdtype(W_coo.data.dtype, np.floating):
        decimals, W_coo.data = _convert_int(W_coo.data)

    # Instantiate a SimpleMinCostFlow solver.
    smcf = min_cost_flow.SimpleMinCostFlow()
    
    # Set up data for smcf
    out_strength = W_coo.sum(axis=1)
    in_strength = W_coo.sum(axis=0).T
    net_strength = out_strength - in_strength
    supplies = net_strength 
    start_nodes = W_coo.row
    end_nodes = W_coo.col
    capacities = W_coo.data
    unit_costs = np.ones(capacities.size)
    
    # Introduce data to smcf
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
    start_nodes, end_nodes, capacities, unit_costs
    )
    smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)  # np.arange give index of nodes (as range(0,n))
    
    # Find the min cost flow.
    status = smcf.solve()
    
    # Check if found optimal
    if status != smcf.OPTIMAL: 
        raise Exception(f"Optimization failed. Status: {status}")
    # smcf.optimal_cost()/10**decimals # If interestee in cost in optimal 
    
    solution_flows = smcf.flows(all_arcs)
    D = coo_array( (solution_flows, (start_nodes, end_nodes)), shape = W.shape)
    
    # Remove zero weight edges and format output 
    D.eliminate_zeros()
    D.data = 10**(-decimals) * D.data
    D = csr_array(D)
    return D 

def _convert_int(array, TOL = 1e-8, MAX = 6):
    """Converst of elements of array into integers by first scaling it by the needed number of decimal places (decimals). 
    
    Parameters
    ----------
    array : np.array [float]
        Array we want to convert.
    TOL : float (default = 1e-8)
        Tolerance to accept that a float corresponds to an integer (after scaling). 
    MAX : int (default = 6)
        Maximal number of decimal places we want to scale array by. 
    
    Returns
    -------
    decimals : int
        Number of decimals places scaled. 
    int_array : np.array [int] 
        Array of integers (after scaling). 
    
    Example
    ---------
    >>> a = 23.34
    >>> convert_int(a) 
    2334
    """
    for decimals in range(MAX+1):
        scaled_array = 10**decimals * array
        int_array = (np.rint(scaled_array)).astype(np.int64) # could use np.int32 for more space efficiency, but with scaling may go over maximum...
        if max(scaled_array - int_array)< TOL:
            return decimals, int_array
    if max(scaled_array) > np.iinfo(np.int64).max:
        raise Exception("Error, integer values have gone over maximum of data type.")
    raise Exception("Error: Decimals not found. You may have to change TOL or you may not be dealing to close to integer data.")