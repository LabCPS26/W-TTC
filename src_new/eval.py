def rank_of_assignment(assignment, pref):
    """
    @brief Return the weak-preference rank of an assignment.
    @param assignment AgentC ID assigned to an AgentE.
    @param pref Weak preference dictionary for one AgentE: {rank: {AgentC IDs}}.
    @return Rank containing the assignment, or None if assignment is unranked.
    """
    for rank, agent_c_ids in pref.items():
        if assignment in agent_c_ids:
            return rank
    return None


def rank_diff(initial_assignment, final_assignment, pref):
    """
    @brief Compute rank improvement from initial assignment to final assignment.
    @param initial_assignment AgentC ID assigned before TTC.
    @param final_assignment AgentC ID assigned after TTC.
    @param pref Weak preference dictionary for one AgentE: {rank: {AgentC IDs}}.
    @return Positive value for improvement, zero for same rank, negative for worse,
        or None if either assignment is unranked.
    """
    initial_rank = rank_of_assignment(initial_assignment, pref)
    final_rank = rank_of_assignment(final_assignment, pref)

    if initial_rank is None or final_rank is None:
        return None

    return initial_rank - final_rank


def all_rank_diffs(initial_match_e, final_match_e, preferences):
    """
    @brief Compute rank improvement for every AgentE in an assignment map.
    @param initial_match_e Mapping from AgentE ID to initial AgentC ID.
    @param final_match_e Mapping from AgentE ID to final AgentC ID.
    @param preferences Mapping from AgentE ID to weak preference dictionary.
    @return Mapping from AgentE ID to rank-improvement details.
    """
    diffs = {}
    for agent_e_id, initial_assignment in sorted(initial_match_e.items()):
        final_assignment = final_match_e.get(agent_e_id, initial_assignment)
        pref = preferences[agent_e_id]
        diffs[agent_e_id] = {
            "initial_assignment": initial_assignment,
            "final_assignment": final_assignment,
            "initial_rank": rank_of_assignment(initial_assignment, pref),
            "final_rank": rank_of_assignment(final_assignment, pref),
            "rank_diff": rank_diff(initial_assignment, final_assignment, pref),
        }
    return diffs


def total_rank_improvement(initial_match_e, final_match_e, preferences):
    """
    @brief Compute total rank improvement over all AgentE IDs.
    @param initial_match_e Mapping from AgentE ID to initial AgentC ID.
    @param final_match_e Mapping from AgentE ID to final AgentC ID.
    @param preferences Mapping from AgentE ID to weak preference dictionary.
    @return Sum of rank improvements, skipping unranked comparisons.
    """
    total_improvement = 0
    for diff in all_rank_diffs(initial_match_e, final_match_e, preferences).values():
        if diff["rank_diff"] is not None:
            total_improvement = total_improvement + diff["rank_diff"]
    return total_improvement


def print_rank_diffs(initial_match_e, final_match_e, preferences):
    """
    @brief Print rank improvements for every AgentE.
    @param initial_match_e Mapping from AgentE ID to initial AgentC ID.
    @param final_match_e Mapping from AgentE ID to final AgentC ID.
    @param preferences Mapping from AgentE ID to weak preference dictionary.
    """
    print("\nRank improvement by AgentE:")
    for agent_e_id, diff in all_rank_diffs(initial_match_e, final_match_e, preferences).items():
        print(
            f"AgentE {agent_e_id}: "
            f"{diff['initial_assignment']} (rank {diff['initial_rank']}) -> "
            f"{diff['final_assignment']} (rank {diff['final_rank']}), "
            f"improvement {diff['rank_diff']}"
        )

    print(f"Total rank improvement: {total_rank_improvement(initial_match_e, final_match_e, preferences)}")
