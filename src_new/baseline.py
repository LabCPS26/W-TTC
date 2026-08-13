from collections import Counter
from copy import deepcopy
from itertools import islice
import random

import networkx as nx

from eval import rank_diff


VALID_FLATTEN_SCHEMES = {"select_first", "select_random"}


def flatten_pref(pref, scheme="select_first", seed=None, initial_assignment=None):
    """
    @brief Convert weak preferences into strict one-value-per-rank preferences.
    @param pref Weak preference dictionary: {rank: {AgentC IDs}}.
    @param scheme Selection scheme for tied values. Supported: "select_first" and
        "select_random".
    @param seed Optional random seed for reproducible "select_random" choices.
    @param initial_assignment Optional initially endowed AgentC ID. If it appears
        in a tied rank, it is selected regardless of scheme.
    @return Flattened preference dictionary: {rank: {one AgentC ID}}.
    @throws ValueError If the scheme is unknown or a rank has no values.
    """
    rng = random.Random(seed)
    return _flatten_pref_with_rng(pref, scheme, rng, initial_assignment)


def flatten_preferences(preferences, scheme="select_first", seed=None, initial_match_e=None):
    """
    @brief Flatten preferences for all AgentE IDs.
    @param preferences Mapping from AgentE ID to weak preference dictionary.
    @param scheme Selection scheme for tied values. Supported: "select_first" and
        "select_random".
    @param seed Optional random seed for reproducible "select_random" choices.
    @param initial_match_e Optional mapping from AgentE ID to initially endowed
        AgentC ID. If an initial assignment appears in a tied rank for that
        AgentE, it is selected regardless of scheme.
    @return Mapping from AgentE ID to flattened preference dictionary.
    """
    if initial_match_e is None:
        initial_match_e = {}

    rng = random.Random(seed)
    return {
        agent_e_id: _flatten_pref_with_rng(
            pref, scheme, rng, initial_match_e.get(agent_e_id)
        )
        for agent_e_id, pref in sorted(preferences.items())
    }


def _flatten_pref_with_rng(pref, scheme, rng, initial_assignment=None):
    """
    @brief Flatten one preference dictionary using an existing random generator.
    @param pref Weak preference dictionary: {rank: {AgentC IDs}}.
    @param scheme Selection scheme for tied values.
    @param rng Random number generator used by "select_random".
    @param initial_assignment Optional initially endowed AgentC ID.
    @return Flattened preference dictionary.
    """
    if scheme not in VALID_FLATTEN_SCHEMES:
        raise ValueError(
            f"Unknown flatten scheme: {scheme}. "
            f"Expected one of {sorted(VALID_FLATTEN_SCHEMES)}."
        )

    flattened_pref = {}
    for rank in sorted(pref):
        agent_c_ids = sorted(pref[rank])
        if not agent_c_ids:
            raise ValueError(f"Rank {rank} has no AgentC IDs to select from.")

        if initial_assignment in agent_c_ids:
            selected_agent_c_id = initial_assignment
        elif scheme == "select_first":
            selected_agent_c_id = agent_c_ids[0]
        else:
            selected_agent_c_id = rng.choice(agent_c_ids)

        flattened_pref[rank] = {selected_agent_c_id}

    return flattened_pref


def ReactTTC_variant(
    agents_e,
    agents_c,
    flatten_scheme="select_first",
    flatten_seed=None,
    debug=False,
    max_rounds=None,
    print_cycle_counts=False,
    cycle_count_progress_interval=10000,
    cycle_batch_size=100,
    return_cycle_count=False,
):
    """
    @brief Run a ReACT-TTC-style baseline using flattened preferences.
    @param agents_e List of AgentE objects with initial assignments and weak preferences.
    @param agents_c List of AgentC objects with capacities and initial assignees.
    @param flatten_scheme Scheme used to flatten weak preferences.
    @param flatten_seed Optional seed for "select_random" preference flattening.
    @param debug If True, print graph vertices, edges, and cycles each round.
    @param max_rounds Optional limit on the number of rounds to execute.
    @param print_cycle_counts If True, print the number of cycles found each round.
    @param cycle_count_progress_interval Cycle-enumeration progress print interval.
    @param cycle_batch_size Number of simple cycles to collect and resolve at a time.
    @param return_cycle_count If True, include the total number of cycles found.
    @return Tuple containing final AgentE-to-AgentC and AgentC-to-AgentE assignment maps.

    This implementation is intentionally separate from weakTTC.py. It first
    flattens each weak preference list into one AgentC per rank, then resolves
    overlapping cycles using the high-gamma rule: cycles are ordered by rank
    improvement summed only over vertices that overlap with other cycles.
    """
    current_match_e, current_match_c = _react_build_assignment_maps(agents_e, agents_c)
    current_match_e = deepcopy(current_match_e)
    current_match_c = deepcopy(current_match_c)

    initial_match_e = {
        agent_e.ID: agent_e.initial_assignment
        for agent_e in agents_e
    }
    weak_preferences = {
        agent_e.ID: agent_e.pref
        for agent_e in agents_e
    }
    flattened_preferences = flatten_preferences(
        weak_preferences,
        scheme=flatten_scheme,
        seed=flatten_seed,
        initial_match_e=initial_match_e,
    )

    participants = {agent_e.ID for agent_e in agents_e}
    final_match_e = {}
    final_match_c = {agent_c.ID: set() for agent_c in agents_c}
    next_rank = {agent_e.ID: 1 for agent_e in agents_e}
    max_rank = {
        agent_e.ID: max(flattened_preferences[agent_e.ID].keys())
        if flattened_preferences[agent_e.ID]
        else 0
        for agent_e in agents_e
    }

    graph = nx.DiGraph()
    graph.add_nodes_from(participants, tag="real")
    next_virtual_id = len(agents_e)
    round_id = 1
    total_cycle_count = 0

    while participants:
        for agent_e_id in list(participants):
            if agent_e_id not in graph:
                graph.add_node(agent_e_id, tag="real")

            if list(graph.successors(agent_e_id)):
                continue

            assigned_agent_c_id = current_match_e[agent_e_id]

            if next_rank[agent_e_id] > max_rank[agent_e_id]:
                final_match_e[agent_e_id] = assigned_agent_c_id
                final_match_c[assigned_agent_c_id].add(agent_e_id)
                current_match_c[assigned_agent_c_id].discard(agent_e_id)
                participants.remove(agent_e_id)
                graph.remove_node(agent_e_id)
                continue

            preferred_agent_c_ids = _react_get_ranked_agent_c_ids(
                flattened_preferences[agent_e_id], next_rank[agent_e_id]
            )
            next_rank[agent_e_id] = next_rank[agent_e_id] + 1

            if assigned_agent_c_id in preferred_agent_c_ids:
                final_match_e[agent_e_id] = assigned_agent_c_id
                final_match_c[assigned_agent_c_id].add(agent_e_id)
                current_match_c[assigned_agent_c_id].discard(agent_e_id)
                participants.remove(agent_e_id)
                graph.remove_node(agent_e_id)
                continue

            for agent_c_id in sorted(preferred_agent_c_ids):
                while (
                    _react_occupied_capacity(current_match_c, final_match_c, agent_c_id)
                    < _react_capacity_by_id(agents_c, agent_c_id)
                ):
                    virtual_node = next_virtual_id
                    next_virtual_id = next_virtual_id + 1
                    graph.add_node(virtual_node, tag="virtual", agent_c=agent_c_id)
                    current_match_e[virtual_node] = agent_c_id
                    current_match_c[agent_c_id].add(virtual_node)

                for holder in current_match_c[agent_c_id]:
                    if holder not in graph:
                        continue
                    graph.add_edge(agent_e_id, holder)
                    if (
                        graph.nodes[holder].get("tag") == "virtual"
                        and graph.out_degree(holder) == 0
                    ):
                        graph.add_edge(holder, agent_e_id)

        if max_rounds is not None and round_id >= max_rounds:
            if debug:
                print(f"Stopping after max_rounds={max_rounds}.")
            break

        round_cycle_count = _react_resolve_simple_cycles_in_batches(
            graph,
            round_id,
            cycle_batch_size,
            current_match_e,
            current_match_c,
            participants,
            final_match_e,
            final_match_c,
            flattened_preferences,
            _react_debug_print_round,
            "ReACT-TTC",
            debug,
            print_cycle_counts,
            cycle_count_progress_interval,
        )
        total_cycle_count = total_cycle_count + round_cycle_count

        source_nodes = [node for node in graph.nodes if graph.in_degree(node) == 0]
        target_nodes = [
            node for node in graph.nodes if graph.nodes[node].get("tag") == "virtual"
        ]
        all_chains = []
        for source_node in source_nodes:
            for target_node in target_nodes:
                all_chains.extend(nx.all_simple_paths(graph, source_node, target_node))

        _react_resolve_all_cycles(
            all_chains,
            current_match_e,
            current_match_c,
            participants,
            final_match_e,
            final_match_c,
            graph,
            flattened_preferences,
        )

        round_id = round_id + 1

    if return_cycle_count:
        return final_match_e, final_match_c, total_cycle_count
    return final_match_e, final_match_c


def TTC(
    agents_e,
    agents_c,
    flatten_scheme="select_first",
    flatten_seed=None,
    debug=False,
    max_rounds=None,
    print_cycle_counts=False,
    cycle_count_progress_interval=10000,
    cycle_batch_size=100,
    return_cycle_count=False,
):
    """
    @brief Run a TTC baseline using flattened preferences and real nodes only.
    @param agents_e List of AgentE objects with initial assignments and weak preferences.
    @param agents_c List of AgentC objects with capacities and initial assignees.
    @param flatten_scheme Scheme used to flatten weak preferences.
    @param flatten_seed Optional seed for "select_random" preference flattening.
    @param debug If True, print graph vertices, edges, and cycles each round.
    @param max_rounds Optional limit on the number of rounds to execute.
    @param print_cycle_counts If True, print the number of cycles found each round.
    @param cycle_count_progress_interval Cycle-enumeration progress print interval.
    @param cycle_batch_size Number of simple cycles to collect and resolve at a time.
    @param return_cycle_count If True, include the total number of cycles found.
    @return Tuple containing final AgentE-to-AgentC and AgentC-to-AgentE assignment maps.

    This baseline follows the ReACT-TTC implementation structure, but it never
    creates virtual nodes and never adds edges involving virtual nodes. Only real
    AgentE nodes and edges between currently active real holders are considered.
    """
    current_match_e, current_match_c = _react_build_assignment_maps(agents_e, agents_c)
    current_match_e = deepcopy(current_match_e)
    current_match_c = deepcopy(current_match_c)

    initial_match_e = {
        agent_e.ID: agent_e.initial_assignment
        for agent_e in agents_e
    }
    weak_preferences = {
        agent_e.ID: agent_e.pref
        for agent_e in agents_e
    }
    flattened_preferences = flatten_preferences(
        weak_preferences,
        scheme=flatten_scheme,
        seed=flatten_seed,
        initial_match_e=initial_match_e,
    )

    participants = {agent_e.ID for agent_e in agents_e}
    final_match_e = {}
    final_match_c = {agent_c.ID: set() for agent_c in agents_c}
    next_rank = {agent_e.ID: 1 for agent_e in agents_e}
    max_rank = {
        agent_e.ID: max(flattened_preferences[agent_e.ID].keys())
        if flattened_preferences[agent_e.ID]
        else 0
        for agent_e in agents_e
    }

    graph = nx.DiGraph()
    graph.add_nodes_from(participants, tag="real")
    round_id = 1
    total_cycle_count = 0

    while participants:
        for agent_e_id in list(participants):
            if agent_e_id not in graph:
                graph.add_node(agent_e_id, tag="real")

            if list(graph.successors(agent_e_id)):
                continue

            assigned_agent_c_id = current_match_e[agent_e_id]

            if next_rank[agent_e_id] > max_rank[agent_e_id]:
                final_match_e[agent_e_id] = assigned_agent_c_id
                final_match_c[assigned_agent_c_id].add(agent_e_id)
                current_match_c[assigned_agent_c_id].discard(agent_e_id)
                participants.remove(agent_e_id)
                graph.remove_node(agent_e_id)
                continue

            preferred_agent_c_ids = _react_get_ranked_agent_c_ids(
                flattened_preferences[agent_e_id], next_rank[agent_e_id]
            )
            next_rank[agent_e_id] = next_rank[agent_e_id] + 1

            if assigned_agent_c_id in preferred_agent_c_ids:
                final_match_e[agent_e_id] = assigned_agent_c_id
                final_match_c[assigned_agent_c_id].add(agent_e_id)
                current_match_c[assigned_agent_c_id].discard(agent_e_id)
                participants.remove(agent_e_id)
                graph.remove_node(agent_e_id)
                continue

            for agent_c_id in sorted(preferred_agent_c_ids):
                for holder in current_match_c[agent_c_id]:
                    if holder not in graph:
                        continue
                    graph.add_edge(agent_e_id, holder)

        if max_rounds is not None and round_id >= max_rounds:
            if debug:
                print(f"Stopping after max_rounds={max_rounds}.")
            break

        round_cycle_count = _react_resolve_simple_cycles_in_batches(
            graph,
            round_id,
            cycle_batch_size,
            current_match_e,
            current_match_c,
            participants,
            final_match_e,
            final_match_c,
            flattened_preferences,
            _ttc_debug_print_round,
            "TTC",
            debug,
            print_cycle_counts,
            cycle_count_progress_interval,
        )
        total_cycle_count = total_cycle_count + round_cycle_count

        round_id = round_id + 1

    if return_cycle_count:
        return final_match_e, final_match_c, total_cycle_count
    return final_match_e, final_match_c


def _react_build_assignment_maps(agents_e, agents_c):
    """
    @brief Build lookup maps for the current initial assignment.
    @param agents_e List of AgentE objects.
    @param agents_c List of AgentC objects.
    @return Tuple containing AgentE-to-AgentC and AgentC-to-AgentE assignment maps.
    """
    match_e = {agent_e.ID: agent_e.initial_assignment for agent_e in agents_e}
    match_c = {agent_c.ID: set(agent_c.assigned) for agent_c in agents_c}
    return match_e, match_c


def _react_resolve_cycle(
    cycle,
    current_match_e,
    current_match_c,
    participants,
    final_match_e,
    final_match_c,
):
    """
    @brief Resolve one cycle or chain by assigning each source to the next holder's object.
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

        if node_u in participants:
            finalized_assignments.append((node_u, agent_c_id))

    for node_u, agent_c_id in finalized_assignments:
        current_match_e[node_u] = agent_c_id
        final_match_e[node_u] = agent_c_id
        final_match_c[agent_c_id].add(node_u)
        participants.remove(node_u)


def _react_resolve_all_cycles(
    all_cycles,
    current_match_e,
    current_match_c,
    participants,
    final_match_e,
    final_match_c,
    graph,
    flattened_preferences,
):
    """
    @brief Resolve non-overlapping cycles, then high-gamma ordered overlapping cycles.
    @param all_cycles List of cycles or chains returned by the graph search.
    @param current_match_e Mapping from AgentE IDs or virtual nodes to AgentC IDs.
    @param current_match_c Mapping from AgentC IDs to sets of AgentE IDs or virtual nodes.
    @param participants Set of active AgentE IDs.
    @param final_match_e Mapping updated with finalized AgentE assignments.
    @param final_match_c Mapping updated with finalized AgentC assignments.
    @param graph Directed graph from which resolved nodes are removed.
    @param flattened_preferences Mapping from AgentE IDs to flattened preferences.
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

    for cycle in nonoverlapping_cycles:
        _react_resolve_cycle(
            cycle,
            current_match_e,
            current_match_c,
            participants,
            final_match_e,
            final_match_c,
        )
        graph.remove_nodes_from(cycle)

    deleted_vertices = set()
    sorted_overlapping_cycles = sorted(
        overlapping_cycles,
        key=lambda cycle: _react_high_gamma_sort_key(
            cycle,
            current_match_e,
            participants,
            flattened_preferences,
            overlapping_vertices,
        ),
    )
    for cycle in sorted_overlapping_cycles:
        if deleted_vertices.intersection(cycle):
            continue
        _react_resolve_cycle(
            cycle,
            current_match_e,
            current_match_c,
            participants,
            final_match_e,
            final_match_c,
        )
        deleted_vertices.update(cycle)
        graph.remove_nodes_from(cycle)


def _react_resolve_simple_cycles_in_batches(
    graph,
    round_id,
    batch_size,
    current_match_e,
    current_match_c,
    participants,
    final_match_e,
    final_match_c,
    flattened_preferences,
    debug_print_round,
    algorithm_label,
    debug=False,
    print_cycle_counts=False,
    progress_interval=10000,
):
    """
    @brief Enumerate simple cycles lazily and resolve them in bounded batches.
    @param graph Directed graph to search.
    @param round_id Current algorithm round number.
    @param batch_size Maximum number of simple cycles to hold in memory at once.
    @param current_match_e Mapping from AgentE IDs or virtual nodes to AgentC IDs.
    @param current_match_c Mapping from AgentC IDs to sets of AgentE IDs or virtual nodes.
    @param participants Set of active AgentE IDs.
    @param final_match_e Mapping updated with finalized AgentE assignments.
    @param final_match_c Mapping updated with finalized AgentC assignments.
    @param flattened_preferences Mapping from AgentE IDs to flattened preferences.
    @param debug_print_round Function used to print graph and cycle-batch details.
    @param algorithm_label Human-readable algorithm name for progress messages.
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
            _react_batched_simple_cycles(graph, batch_size),
            [],
        )
        if not cycle_batch:
            break

        cycle_batch = sorted(cycle_batch, key=_react_path_sort_key)
        cycle_count = cycle_count + len(cycle_batch)

        if debug:
            debug_print_round(round_id, graph, cycle_batch)

        _react_resolve_all_cycles(
            cycle_batch,
            current_match_e,
            current_match_c,
            participants,
            final_match_e,
            final_match_c,
            graph,
            flattened_preferences,
        )

        if (
            print_cycle_counts
            and progress_interval is not None
            and progress_interval > 0
            and cycle_count % progress_interval == 0
        ):
            print(
                f"    {algorithm_label} round {round_id}: resolved {cycle_count} cycles...",
                flush=True,
            )

    if print_cycle_counts:
        print(
            f"    {algorithm_label} round {round_id}: total cycles resolved = {cycle_count}",
            flush=True,
        )

    return cycle_count


def _react_batched_simple_cycles(graph, batch_size):
    """
    @brief Yield simple cycles in batches without materializing the full cycle set.
    @param graph Directed graph to search.
    @param batch_size Maximum number of cycles per yielded batch.
    @return Iterator of cycle batches.
    """
    cycles = nx.simple_cycles(graph)

    while batch := list(islice(cycles, batch_size)):
        yield batch


def _react_get_ranked_agent_c_ids(flattened_pref, rank):
    """
    @brief Return the AgentC ID set at a given flattened preference rank.
    @param flattened_pref Flattened preference dictionary for one AgentE.
    @param rank Preference rank to read.
    @return Set of AgentC IDs at the requested rank.
    """
    return set(flattened_pref.get(rank, set()))


def _react_high_gamma_sort_key(
    cycle,
    current_match_e,
    participants,
    flattened_preferences,
    overlapping_vertices,
):
    """
    @brief Sort overlapping cycles by high-gamma score from high to low.
    @param cycle Cycle or chain represented as a list of graph nodes.
    @param current_match_e Mapping from AgentE IDs or virtual nodes to AgentC IDs.
    @param participants Set of active AgentE IDs.
    @param flattened_preferences Mapping from AgentE IDs to flattened preferences.
    @param overlapping_vertices Set of vertices appearing in more than one cycle.
    @return Tuple usable as a Python sort key.
    """
    return (
        -_react_cycle_rank_diff_sum(
            cycle,
            current_match_e,
            participants,
            flattened_preferences,
            overlapping_vertices,
        ),
        len(cycle),
        tuple(repr(node) for node in cycle),
    )


def _react_cycle_rank_diff_sum(
    cycle,
    current_match_e,
    participants,
    flattened_preferences,
    scored_vertices,
):
    """
    @brief Compute rank improvement summed over selected real vertices in a cycle.
    @param cycle Cycle or chain represented as a list of graph nodes.
    @param current_match_e Mapping from AgentE IDs or virtual nodes to AgentC IDs.
    @param participants Set of active AgentE IDs.
    @param flattened_preferences Mapping from AgentE IDs to flattened preferences.
    @param scored_vertices Set of vertices allowed to contribute to the score.
    @return Sum of rank improvements for selected AgentE nodes in the cycle.
    """
    total_rank_diff = 0
    for i, node_u in enumerate(cycle):
        if node_u not in participants or node_u not in scored_vertices:
            continue

        node_v = cycle[(i + 1) % len(cycle)]
        current_assignment = current_match_e[node_u]
        cycle_assignment = current_match_e[node_v]
        diff = rank_diff(
            current_assignment,
            cycle_assignment,
            flattened_preferences[node_u],
        )
        if diff is not None:
            total_rank_diff = total_rank_diff + diff

    return total_rank_diff


def _react_capacity_by_id(agents_c, agent_c_id):
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


def _react_occupied_capacity(current_match_c, final_match_c, agent_c_id):
    """
    @brief Count active and finalized assignments occupying an AgentC capacity.
    @param current_match_c Active AgentC holders still participating in the graph.
    @param final_match_c Finalized AgentC assignments removed from the graph.
    @param agent_c_id ID of the AgentC whose occupied capacity is requested.
    @return Number of occupied slots for the requested AgentC.
    """
    return len(current_match_c[agent_c_id]) + len(final_match_c[agent_c_id])


def _react_path_sort_key(path):
    """
    @brief Build a deterministic sort key for paths containing mixed node types.
    @param path Cycle or chain represented as a list of graph nodes.
    @return Tuple usable as a Python sort key.
    """
    return len(path), tuple(repr(node) for node in path)


def _react_debug_print_round(round_id, graph, all_cycles):
    """
    @brief Print the current ReACT-TTC graph and cycles for one algorithm round.
    @param round_id Round number being printed.
    @param graph Directed graph to inspect.
    @param all_cycles Cycles discovered in the graph for this round.
    """
    print(f"\n=== ReACT-TTC Variant Round {round_id} ===")
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


def _ttc_debug_print_round(round_id, graph, all_cycles):
    """
    @brief Print the current TTC graph and cycles for one algorithm round.
    @param round_id Round number being printed.
    @param graph Directed graph to inspect.
    @param all_cycles Cycles discovered in the graph for this round.
    """
    print(f"\n=== TTC Round {round_id} ===")
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
