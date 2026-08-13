from collections import Counter
from copy import deepcopy
from itertools import islice
import random

import networkx as nx

from eval import rank_diff


DEBUG = False
VALID_CYCLE_SORT_SCHEMES = {"shortest", "longest", "rank_diff_sum", "random", "high_gamma"}


def build_assignment_maps(agents_e, agents_c):
    """
    @brief Build lookup maps for the current initial assignment.
    @param agents_e List of AgentE objects.
    @param agents_c List of AgentC objects.
    @return Tuple containing AgentE-to-AgentC and AgentC-to-AgentE assignment maps.
    """
    match_e = {agent_e.ID: agent_e.initial_assignment for agent_e in agents_e}
    match_c = {agent_c.ID: set(agent_c.assigned) for agent_c in agents_c}
    return match_e, match_c


def resolve_cycle(cycle, current_match_e, current_match_c, participants, final_match_e, final_match_c):
    """
    @brief Resolve one TTC cycle or chain by assigning each source to the next holder's object.
    @param cycle Ordered list of graph nodes in the cycle or chain.
    @param current_match_e Mapping from AgentE IDs or virtual nodes to AgentC IDs.
    @param current_match_c Mapping from AgentC IDs to sets of AgentE IDs or virtual nodes.
    @param participants Set of active AgentE IDs.
    @param final_match_e Mapping updated with finalized AgentE assignments.
    @param final_match_c Mapping updated with finalized AgentC assignments.
    """
    finalized_assignments = []

    for i, node_u in enumerate(cycle):
        node_v = cycle[(i + 1) % len(cycle)]
        agent_c_id = current_match_e[node_v]

        if node_v in current_match_c[agent_c_id]:
            current_match_c[agent_c_id].remove(node_v)

        if node_u in participants: # This check is to avoid processing virtual nodes that are not in participants but can be in cycle
            finalized_assignments.append((node_u, agent_c_id))

    for node_u, agent_c_id in finalized_assignments:
        current_match_e[node_u] = agent_c_id # It might not be required to update current_match_e, but we do it for consistency
        final_match_e[node_u] = agent_c_id
        final_match_c[agent_c_id].add(node_u)
        participants.remove(node_u)


def resolve_all_cycles(
    all_cycles,
    current_match_e,
    current_match_c,
    participants,
    final_match_e,
    final_match_c,
    graph,
    preferences,
    cycle_sort_scheme="rank_diff_sum",
    cycle_sort_rng=None,
):
    """
    @brief Resolve all non-overlapping cycles, then selected overlapping cycles.
    @param all_cycles List of cycles or chains returned by the TTC graph search.
    @param current_match_e Mapping from AgentE IDs or virtual nodes to AgentC IDs.
    @param current_match_c Mapping from AgentC IDs to sets of AgentE IDs or virtual nodes.
    @param participants Set of active AgentE IDs.
    @param final_match_e Mapping updated with finalized AgentE assignments.
    @param final_match_c Mapping updated with finalized AgentC assignments.
    @param graph Directed graph from which resolved nodes are removed.
    @param preferences Mapping from AgentE IDs to weak preference dictionaries.
    @param cycle_sort_scheme Scheme used to order overlapping cycles.
    @param cycle_sort_rng Random number generator used by the "random" scheme.
    """
    if not all_cycles:
        return

    vertices_in_cycles = [node for cycle in all_cycles for node in cycle]
    overlapping_vertices = {
        node for node, count in Counter(vertices_in_cycles).items() if count > 1
    }

    nonoverlapping_cycles = [
        cycle for cycle in all_cycles if not overlapping_vertices.intersection(cycle)
    ]
    overlapping_cycles = [
        cycle for cycle in all_cycles if cycle not in nonoverlapping_cycles
    ]

    # Resolve non-overlapping cycles 
    for cycle in nonoverlapping_cycles:
        resolve_cycle(cycle, current_match_e, current_match_c, participants, final_match_e, final_match_c)
        graph.remove_nodes_from(cycle)

    # Resolve overlapping cycles in sorted order, skipping any that share vertices with already-resolved cycles
    deleted_vertices = set()
    sorted_overlapping_cycles = _sort_overlapping_cycles(
        overlapping_cycles,
        current_match_e,
        participants,
        preferences,
        cycle_sort_scheme,
        cycle_sort_rng,
        overlapping_vertices,
    )
    for cycle in sorted_overlapping_cycles:
        if deleted_vertices.intersection(cycle): # Skip cycles that share vertices with already-resolved cycles
            continue
        resolve_cycle(cycle, current_match_e, current_match_c, participants, final_match_e, final_match_c)
        deleted_vertices.update(cycle)
        graph.remove_nodes_from(cycle)


def get_ranked_agent_c_ids(agent_e, rank):
    """
    @brief Return the AgentC IDs tied at a given weak-preference rank.
    @param agent_e AgentE object whose preference dictionary is queried.
    @param rank Weak-preference rank to read.
    @return Set of AgentC IDs at the requested rank.
    """
    return set(agent_e.pref.get(rank, set()))


def run_weak_ttc(
    agents_e,
    agents_c,
    debug=DEBUG,
    max_rounds=None,
    cycle_sort_scheme="rank_diff_sum",
    cycle_sort_seed=None,
    print_cycle_counts=False,
    cycle_count_progress_interval=10000,
    cycle_batch_size=100,
    return_cycle_count=False,
):
    """
    @brief Run a TTC-style algorithm using weak preferences over AgentC objects.
    @param agents_e List of AgentE objects with initial assignments and weak preferences.
    @param agents_c List of AgentC objects with capacities and initial assignees.
    @param debug If True, print graph vertices, edges, and cycles each round.
    @param max_rounds Optional limit on the number of rounds to execute.
    @param cycle_sort_scheme Scheme used to resolve overlapping cycles. Supported:
        "rank_diff_sum", "shortest", "longest", "random", and "high_gamma".
    @param cycle_sort_seed Optional random seed for the "random" cycle sort scheme.
    @param print_cycle_counts If True, print the number of cycles found each round.
    @param cycle_count_progress_interval Cycle-enumeration progress print interval.
    @param cycle_batch_size Number of simple cycles to collect and resolve at a time.
    @param return_cycle_count If True, include the total number of cycles found.
    @return Tuple containing final AgentE-to-AgentC and AgentC-to-AgentE assignment maps.

    Each active AgentE points to every current holder of the AgentC IDs in its
    next weak-preference rank. If an AgentC has open capacity, a virtual node is
    inserted as a holder for that capacity. Cycles and chains ending at virtual
    nodes are then resolved.
    """
    if cycle_sort_scheme not in VALID_CYCLE_SORT_SCHEMES:
        raise ValueError(
            f"Unknown cycle_sort_scheme: {cycle_sort_scheme}. "
            f"Expected one of {sorted(VALID_CYCLE_SORT_SCHEMES)}."
        )

    agents_e_by_id = {agent_e.ID: agent_e for agent_e in agents_e}
    preferences = {agent_e.ID: agent_e.pref for agent_e in agents_e}
    cycle_sort_rng = random.Random(cycle_sort_seed)
    current_match_e, current_match_c = build_assignment_maps(agents_e, agents_c)
    current_match_e = deepcopy(current_match_e)
    current_match_c = deepcopy(current_match_c)

    participants = {agent_e.ID for agent_e in agents_e}
    final_match_e = {}
    final_match_c = {agent_c.ID: set() for agent_c in agents_c}
    next_rank = {agent_e.ID: 1 for agent_e in agents_e}
    max_rank = {
        agent_e.ID: max(agent_e.pref.keys()) if agent_e.pref else 0
        for agent_e in agents_e
    }

    graph = nx.DiGraph()
    graph.add_nodes_from(participants, tag="real")
    n_e = len(agents_e)
    next_virtual_id = n_e
    round_id = 1
    total_cycle_count = 0

    while participants:
        for agent_e_id in list(participants):

            # AK: Most probably we don't need it
            if agent_e_id not in graph:
                graph.add_node(agent_e_id, tag="real")

            # If there is any outgoing  edge for the agent_e, skip adding new preference edges for this agent in this iteration
            if list(graph.successors(agent_e_id)):
                continue

            assigned_agent_c_id = current_match_e[agent_e_id]
            
            # AK: We don't need it
            if next_rank[agent_e_id] > max_rank[agent_e_id]:
                final_match_e[agent_e_id] = assigned_agent_c_id
                final_match_c[assigned_agent_c_id].add(agent_e_id)
                current_match_c[assigned_agent_c_id].discard(agent_e_id)
                participants.remove(agent_e_id)
                graph.remove_node(agent_e_id)
                continue

            preferred_agent_c_ids = get_ranked_agent_c_ids(
                agents_e_by_id[agent_e_id], next_rank[agent_e_id]
            )
            next_rank[agent_e_id] = next_rank[agent_e_id] + 1

            # If the current assignment is in the preferred set, we can assign it and remove the agent from the graph and participants
            if assigned_agent_c_id in preferred_agent_c_ids:
                final_match_e[agent_e_id] = assigned_agent_c_id
                final_match_c[assigned_agent_c_id].add(agent_e_id)
                current_match_c[assigned_agent_c_id].discard(agent_e_id)
                participants.remove(agent_e_id)
                graph.remove_node(agent_e_id)
                continue

            

            for agent_c_id in sorted(preferred_agent_c_ids):
                # Add virtual nodes for any AgentC that has open capacity
                while _occupied_capacity(current_match_c, final_match_c, agent_c_id) < _capacity_by_id(agents_c, agent_c_id):
                    virtual_node = next_virtual_id
                    next_virtual_id = next_virtual_id + 1
                    graph.add_node(virtual_node, tag="virtual", agent_c=agent_c_id)
                    current_match_e[virtual_node] = agent_c_id
                    current_match_c[agent_c_id].add(virtual_node)

                for holder in current_match_c[agent_c_id]:
                    if holder not in graph:
                        continue
                    graph.add_edge(agent_e_id, holder) # add edge from the agent_e to the preferred agent node
                    if (
                        graph.nodes[holder].get("tag") == "virtual"
                        and graph.out_degree(holder) == 0
                    ):  ##AK: added extra restriction of checking outdegree 0 to avoid increasing the execution time exponentially. It decreses the solution quality a little bit, but makes the sol faster 
                        graph.add_edge(holder, agent_e_id) # add edge from the virtual node to agent_e if it has no outgoing edges
                        """
                        ****Issue:****
                        ####AK:****Check here. we are not adding edge to all agents and now if some agent v has edge to the virtual node, but let previously virtual node was pointing to u and u is
                        now removed in the last round, then v will be skipped as it has outgoing edge and the virtual node will never point anything back. 
                        This is why the execution fails without the chain handling part 
                        """


        if max_rounds is not None and round_id >= max_rounds:
            if debug:
                print(f"Stopping after max_rounds={max_rounds}.")
            break
        
        ## AK: Write about batch processing in the paper. It is important to avoid memory issues and also to avoid increasing the execution time.
        round_cycle_count = _resolve_simple_cycles_in_batches(
            graph,
            round_id,
            cycle_batch_size,
            current_match_e,
            current_match_c,
            participants,
            final_match_e,
            final_match_c,
            preferences,
            cycle_sort_scheme,
            cycle_sort_rng,
            debug,
            print_cycle_counts,
            cycle_count_progress_interval,
        )
        total_cycle_count = total_cycle_count + round_cycle_count

        ## AK: Check if we need the below part
        ## Chain handling. Currently required.
        source_nodes = [node for node in graph.nodes if graph.in_degree(node) == 0]
        target_nodes = [
            node for node in graph.nodes if graph.nodes[node].get("tag") == "virtual"
        ]
        all_chains = []
        for source_node in source_nodes:
            for target_node in target_nodes:
                all_chains.extend(nx.all_simple_paths(graph, source_node, target_node))

        resolve_all_cycles(
            all_chains,
            current_match_e,
            current_match_c,
            participants,
            final_match_e,
            final_match_c,
            graph,
            preferences,
            cycle_sort_scheme,
            cycle_sort_rng,
        )

        round_id = round_id + 1

    if return_cycle_count:
        return final_match_e, final_match_c, total_cycle_count
    return final_match_e, final_match_c


def _resolve_simple_cycles_in_batches(
    graph,
    round_id,
    batch_size,
    current_match_e,
    current_match_c,
    participants,
    final_match_e,
    final_match_c,
    preferences,
    cycle_sort_scheme,
    cycle_sort_rng,
    debug=False,
    print_cycle_counts=False,
    progress_interval=10000,
):
    """
    @brief Enumerate simple cycles lazily and resolve them in bounded batches.
    @param graph Directed graph to search.
    @param round_id Current WeakTTC round number.
    @param batch_size Maximum number of simple cycles to hold in memory at once.
    @param current_match_e Mapping from AgentE IDs or virtual nodes to AgentC IDs.
    @param current_match_c Mapping from AgentC IDs to sets of AgentE IDs or virtual nodes.
    @param participants Set of active AgentE IDs.
    @param final_match_e Mapping updated with finalized AgentE assignments.
    @param final_match_c Mapping updated with finalized AgentC assignments.
    @param preferences Mapping from AgentE IDs to weak preference dictionaries.
    @param cycle_sort_scheme Scheme used to order overlapping cycles.
    @param cycle_sort_rng Random number generator used by the "random" scheme.
    @param debug If True, print graph vertices, edges, and cycle batches.
    @param print_cycle_counts If True, print cycle counts.
    @param progress_interval Number of cycles between progress prints.
    @return Number of simple cycles enumerated and resolved.
    """
    if batch_size is None or batch_size <= 0:
        raise ValueError("cycle_batch_size must be a positive integer.")

    cycle_count = 0

    while True:
        cycle_batch = next(
            _batched_simple_cycles(graph, batch_size),
            [],
        )
        if not cycle_batch:
            break

        cycle_batch = sorted(cycle_batch, key=_path_sort_key)
        cycle_count = cycle_count + len(cycle_batch)

        if debug:
            _debug_print_round(round_id, graph, cycle_batch)

        resolve_all_cycles(
            cycle_batch,
            current_match_e,
            current_match_c,
            participants,
            final_match_e,
            final_match_c,
            graph,
            preferences,
            cycle_sort_scheme,
            cycle_sort_rng,
        )

        if (
            print_cycle_counts
            and progress_interval is not None
            and progress_interval > 0
            and cycle_count % progress_interval == 0
        ):
            print(
                f"    WeakTTC round {round_id}: resolved {cycle_count} cycles...",
                flush=True,
            )

    if print_cycle_counts:
        print(
            f"    WeakTTC round {round_id}: total cycles resolved = {cycle_count}",
            flush=True,
        )

    return cycle_count


def _batched_simple_cycles(graph, batch_size):
    """
    @brief Yield simple cycles in batches without materializing the full cycle set.
    @param graph Directed graph to search.
    @param batch_size Maximum number of cycles per yielded batch.
    @return Iterator of cycle batches.
    """
    cycles = nx.simple_cycles(graph)

    while batch := list(islice(cycles, batch_size)):
        yield batch


def _capacity_by_id(agents_c, agent_c_id):
    """
    @brief Look up the capacity of an AgentC by ID.
    @param agents_c List of AgentC objects.
    @param agent_c_id ID of the AgentC whose capacity is requested.
    @return Capacity of the requested AgentC.
    @throws KeyError If no AgentC has the requested ID.
    """
    for agent_c in agents_c:
        if agent_c.ID == agent_c_id:
            return agent_c.capacity
    raise KeyError(f"Unknown AgentC ID: {agent_c_id}")


def _occupied_capacity(current_match_c, final_match_c, agent_c_id):
    """
    @brief Count active and finalized assignments occupying an AgentC capacity.
    @param current_match_c Active AgentC holders still participating in the graph.
    @param final_match_c Finalized AgentC assignments removed from the graph.
    @param agent_c_id ID of the AgentC whose occupied capacity is requested.
    @return Number of occupied slots for the requested AgentC.
    """
    return len(current_match_c[agent_c_id]) + len(final_match_c[agent_c_id])


def _path_sort_key(path):
    """
    @brief Build a deterministic sort key for paths containing mixed node types.
    @param path Cycle or chain represented as a list of graph nodes.
    @return Tuple usable as a Python sort key.
    """
    return len(path), tuple(repr(node) for node in path)


def _sort_overlapping_cycles(
    overlapping_cycles,
    current_match_e,
    participants,
    preferences,
    cycle_sort_scheme,
    cycle_sort_rng,
    overlapping_vertices,
):
    """
    @brief Order overlapping cycles according to the selected resolution scheme.
    @param overlapping_cycles List of overlapping cycles or chains.
    @param current_match_e Mapping from AgentE IDs or virtual nodes to AgentC IDs.
    @param participants Set of active AgentE IDs.
    @param preferences Mapping from AgentE IDs to weak preference dictionaries.
    @param cycle_sort_scheme Scheme used to order overlapping cycles.
    @param cycle_sort_rng Random number generator used by the "random" scheme.
    @param overlapping_vertices Set of vertices appearing in more than one cycle.
    @return Ordered list of overlapping cycles.
    """
    if cycle_sort_scheme == "random":
        ordered_cycles = sorted(overlapping_cycles, key=_path_sort_key)
        cycle_sort_rng.shuffle(ordered_cycles)
        return ordered_cycles

    return sorted(
        overlapping_cycles,
        key=lambda cycle: _cycle_sort_key(
            cycle,
            current_match_e,
            participants,
            preferences,
            cycle_sort_scheme,
            overlapping_vertices,
        ),
    )


def _cycle_sort_key(
    cycle,
    current_match_e,
    participants,
    preferences,
    cycle_sort_scheme,
    overlapping_vertices,
):
    """
    @brief Build a sort key for overlapping cycle resolution.
    @param cycle Cycle or chain represented as a list of graph nodes.
    @param current_match_e Mapping from AgentE IDs or virtual nodes to AgentC IDs.
    @param participants Set of active AgentE IDs.
    @param preferences Mapping from AgentE IDs to weak preference dictionaries.
    @param cycle_sort_scheme Scheme used to order overlapping cycles.
    @param overlapping_vertices Set of vertices appearing in more than one cycle.
    @return Tuple usable as a Python sort key.
    """
    tie_breaker = tuple(repr(node) for node in cycle)

    if cycle_sort_scheme == "shortest":
        return len(cycle), tie_breaker

    if cycle_sort_scheme == "longest":
        return -len(cycle), tie_breaker

    if cycle_sort_scheme == "rank_diff_sum":
        return -_cycle_rank_diff_sum(cycle, current_match_e, participants, preferences), len(cycle), tie_breaker

    if cycle_sort_scheme == "high_gamma":
        return -_cycle_rank_diff_sum(
            cycle, current_match_e, participants, preferences, overlapping_vertices
        ), len(cycle), tie_breaker

    raise ValueError(
        f"Unknown cycle_sort_scheme: {cycle_sort_scheme}. "
        f"Expected one of {sorted(VALID_CYCLE_SORT_SCHEMES)}."
    )


def _cycle_rank_diff_sum(cycle, current_match_e, participants, preferences, scored_vertices=None):
    """
    @brief Compute the total rank improvement if one cycle were resolved.
    @param cycle Cycle or chain represented as a list of graph nodes.
    @param current_match_e Mapping from AgentE IDs or virtual nodes to AgentC IDs.
    @param participants Set of active AgentE IDs.
    @param preferences Mapping from AgentE IDs to weak preference dictionaries.
    @param scored_vertices Optional set limiting which vertices contribute to the sum.
    @return Sum of rank improvements for real AgentE nodes in the cycle.
    """
    total_rank_diff = 0
    for i, node_u in enumerate(cycle):
        if node_u not in participants:
            continue
        if scored_vertices is not None and node_u not in scored_vertices:
            continue

        node_v = cycle[(i + 1) % len(cycle)]
        current_assignment = current_match_e[node_u]
        cycle_assignment = current_match_e[node_v]
        diff = rank_diff(current_assignment, cycle_assignment, preferences[node_u])
        if diff is not None:
            total_rank_diff = total_rank_diff + diff

    return total_rank_diff


def _debug_print_round(round_id, graph, all_cycles):
    """
    @brief Print the current TTC graph and cycles for one algorithm round.
    @param round_id Round number being printed.
    @param graph Directed graph to inspect.
    @param all_cycles Cycles discovered in the graph for this round.
    """
    print(f"\n=== Weak TTC Round {round_id} ===")
    print("Vertices:")
    for node, data in sorted(graph.nodes(data=True), key=lambda item: repr(item[0])):
        print(f"  {node}: {dict(sorted(data.items()))}")

    print("Edges:")
    for source, target in sorted(graph.edges(), key=lambda edge: (repr(edge[0]), repr(edge[1]))):
        print(f"  {source} -> {target}")

    print("Found cycles:")
    if not all_cycles:
        print("  None")
        return

    for cycle in all_cycles:
        print(f"  {cycle}")
