import numpy as np
import igraph as ig
from collections import defaultdict

#################################################
### 1. METRICHE TOPOLOGICHE GLOBALI (GRAPH-LEVEL)
#################################################

def calculate_global_graph_metrics(g):
    """
    Calcola metriche globali di teoria dei grafi sull'intero grafo vascolare.
    
    :param g: Oggetto grafo igraph (deve essere non diretto)
    :return: Dizionario di metriche globali
    """
    metrics = {}
    
    # Assicura che il grafo sia non diretto per queste metriche
    if g.is_directed():
        g = g.as_undirected()

    # Efficienza Globale (basata sulla lunghezza media del percorso minimo)
    # Nota: igraph calcola la 'distanza media', l'efficienza è l'inverso.
    # Questo può essere MOLTO lento su grafi grandi e densi.
    try:
        avg_path_len = g.average_path_length()
        metrics['average_path_length'] = avg_path_len
        metrics['global_efficiency'] = 1 / avg_path_len if avg_path_len > 0 else 0
    except Exception as e:
        print(f"Attenzione: Impossibile calcolare l'efficienza globale (grafo grande?): {e}")
        metrics['average_path_length'] = np.nan
        metrics['global_efficiency'] = np.nan

    # Coefficiente di Clustering Globale (Transitivity)
    # Misura la tendenza dei nodi a raggrupparsi.
    metrics['clustering_coefficient'] = g.transitivity_undirected()
    
    # Grado Medio dei Nodi
    metrics['average_node_degree'] = np.mean(g.degree())
    
    return metrics

def calculate_fractal_dimension(points, n_boxes=10):
    """
    Calcola la Dimensione Frattale (Box-Counting) da un set di punti 3D.
    
    :param points: Array Numpy di coordinate (N, 3)
    :return: Dimensione frattale stimata
    """
    if points.shape[0] < 10:
        return np.nan # Dati insufficienti

    # Trova i limiti del volume
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # La dimensione massima del box è la dimensione massima del volume
    max_size = (max_coords - min_coords).max()
    if max_size == 0:
        return 0.0

    # Crea una serie di dimensioni dei box, logaritmicamente
    sizes = np.logspace(np.log10(max_size / n_boxes), np.log10(max_size), n_boxes)
    
    counts = []
    for size in sizes:
        if size < 1e-6: continue
            
        # Crea la griglia di box
        bins = [np.arange(min_coords[i], max_coords[i] + size, size) for i in range(3)]
        
        # Istogramma 3D: conta quanti punti cadono in ogni box
        H, edges = np.histogramdd(points, bins=bins)
        
        # Conta il numero di box non vuoti
        counts.append(np.sum(H > 0))

    if len(counts) < 2:
        return np.nan # Non abbastanza dati per il fit

    # Calcola il fit lineare log-log
    # log(count) = D * log(1/size) + C
    # log(count) = -D * log(size) + C
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dimension = -coeffs[0] # La pendenza è -D
    
    return fractal_dimension


#################################################
### 2. METRICHE DI DIRAMAZIONE (NODE-LEVEL)
#################################################

def get_vector(g, v_idx_1, v_idx_2, resolution):
    """Ottiene un vettore normalizzato tra due vertici."""
    p1 = np.array(g.vs[v_idx_1]['v_coords']) * resolution
    p2 = np.array(g.vs[v_idx_2]['v_coords']) * resolution
    vec = p2 - p1
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def calculate_branching_metrics(g, resolution):
    """
    Calcola metriche alle biforcazioni (grado 3).
    Fa un'ipotesi semplificata sul "genitore" (vaso più spesso).
    
    :param g: Oggetto grafo igraph
    :param resolution: Risoluzione [x, y, z]
    :return: Liste di deviazioni di Murray, angoli e rapporti di asimmetria
    """
    murray_deviations = []
    branching_angles_1 = [] # Angolo tra genitore e figlio 1
    branching_angles_2 = [] # Angolo tra genitore e figlio 2
    asymmetry_ratios = []
    
    # Trova tutti i punti di diramazione (biforcazioni)
    branch_points = g.vs.select(_degree_eq=3)
    
    for bp in branch_points:
        bp_idx = bp.index
        neighbor_indices = g.neighbors(bp_idx)
        
        if len(neighbor_indices) != 3:
            continue # Dovrebbe essere 3, ma per sicurezza

        # Ottieni i raggi e i vettori dei 3 segmenti connessi
        # Usiamo il raggio medio dei primi 2 punti del segmento come stima
        radii = []
        vectors = []
        for n_idx in neighbor_indices:
            # Stima del raggio: media del raggio del nodo e del suo vicino
            r = (g.vs[bp_idx]['v_radius'] + g.vs[n_idx]['v_radius']) / 2.0
            radii.append(r)
            vectors.append(get_vector(g, bp_idx, n_idx, resolution))
            
        radii = np.array(radii)
        vectors = np.array(vectors)
        
        # Ipotesi: il "genitore" è il segmento con il raggio più grande
        parent_idx = np.argmax(radii)
        child_indices = [i for i in range(3) if i != parent_idx]
        
        if len(child_indices) != 2:
            continue
            
        r_p = radii[parent_idx]
        r_c1 = radii[child_indices[0]]
        r_c2 = radii[child_indices[1]]
        
        v_p = vectors[parent_idx] * -1 # Vettore punta *verso* il nodo
        v_c1 = vectors[child_indices[0]]
        v_c2 = vectors[child_indices[1]]

        # a) Legge di Murray
        if r_p > 0:
            murray_dev = (r_c1**3 + r_c2**3) / r_p**3
            murray_deviations.append(murray_dev)
        
        # b) Rapporto di Asimmetria
        if max(r_c1, r_c2) > 0:
            asym_ratio = min(r_c1, r_c2) / max(r_c1, r_c2)
            asymmetry_ratios.append(asym_ratio)
            
        # c) Angoli di Diramazione (in gradi)
        # Usiamo il prodotto scalare: A · B = ||A|| ||B|| cos(theta)
        # Poiché i vettori sono normalizzati: cos(theta) = A · B
        # L'angolo è arccos(A · B)
        cos_theta1 = np.dot(v_p, v_c1)
        cos_theta2 = np.dot(v_p, v_c2)
        
        # Limita i valori a [-1, 1] per evitare errori di arrotondamento
        angle1 = np.degrees(np.arccos(np.clip(cos_theta1, -1.0, 1.0)))
        angle2 = np.degrees(np.arccos(np.clip(cos_theta2, -1.0, 1.0)))
        
        branching_angles_1.append(angle1)
        branching_angles_2.append(angle2)

    return {
        'murray_deviations': murray_deviations,
        'branching_angles_child1': branching_angles_1,
        'branching_angles_child2': branching_angles_2,
        'asymmetry_ratios': asymmetry_ratios
    }