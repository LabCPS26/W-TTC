from time import perf_counter

from agents import initialize_agents, initialize_preferences, set_initial_endowment
from baseline import ReactTTC_variant, TTC
from eval import print_rank_diffs, total_rank_improvement
from weakTTC import run_weak_ttc


DEBUG = False



def debug_print_results(
    agents_e,
    agents_c,
    initial_endowment,
    weak_final_match_e,
    weak_final_match_c,
    ttc_final_match_e,
    ttc_final_match_c,
    react_final_match_e,
    react_final_match_c,
    preferences,
):
    """
    @brief Print detailed initialized state and matching results when DEBUG is enabled.
    @param agents_e List of AgentE objects.
    @param agents_c List of AgentC objects.
    @param initial_endowment Initial AgentE-to-AgentC assignment map.
    @param weak_final_match_e WeakTTC final AgentE-to-AgentC assignment map.
    @param weak_final_match_c WeakTTC final AgentC-to-AgentE assignment map.
    @param ttc_final_match_e TTC final AgentE-to-AgentC assignment map.
    @param ttc_final_match_c TTC final AgentC-to-AgentE assignment map.
    @param react_final_match_e ReACT-TTC final AgentE-to-AgentC assignment map.
    @param react_final_match_c ReACT-TTC final AgentC-to-AgentE assignment map.
    @param preferences Mapping from AgentE IDs to weak preference dictionaries.
    """
    print("Initialized AgentE:")
    for agent_e in agents_e:
        print(agent_e)

    print("\nInitialized AgentC:")
    for agent_c in agents_c:
        print(agent_c)

    print("\nInitial endowment:", initial_endowment)
    print("Weak TTC final AgentE matching:", weak_final_match_e)
    print("Weak TTC final AgentC matching:", weak_final_match_c)
    print("TTC final AgentE matching:", ttc_final_match_e)
    print("TTC final AgentC matching:", ttc_final_match_c)
    print("ReACT-TTC final AgentE matching:", react_final_match_e)
    print("ReACT-TTC final AgentC matching:", react_final_match_c)
    print_rank_diffs(initial_endowment, weak_final_match_e, preferences)
    print("Preferences written to src_new/preferences.json")


def main():
    """
    @brief Initialize a small example problem instance.

    Creates AgentE and AgentC objects, assigns the initial endowment,
    generates weak preferences, writes preferences to JSON, and prints
    the initialized state.
    """
    n_e = 55
    n_c = 30
    capacity = 2
    capacity_type="strict" # "strict" uses capacity for every agentC; "loose" randomly assigns each AgentC capacity between 1 and capacity.
    max_rank_size = 2   # Number of maximum AgentC IDs that may share one rank in weak preferences.
    seed = 123
    # cycle_sort_scheme Supported: "rank_diff_sum", "shortest", "longest", "random", and "high_gamma".
    cycle_sort_scheme = "rank_diff_sum"

    print(f"n_e: {n_e}")
    print(f"n_c: {n_c}")
    print(f"capacity: {capacity}")
    print(f"max_rank_size: {max_rank_size}")

    agents_e, agents_c = initialize_agents(n_e=n_e, n_c=n_c, capacity=capacity, capacity_type=capacity_type, seed=seed)
    initial_endowment = set_initial_endowment(agents_e, agents_c, seed=seed)
    preferences = initialize_preferences(agents_e, agents_c, max_rank_size=max_rank_size, seed=seed)
    weak_start = perf_counter()
    weak_final_match_e, weak_final_match_c = run_weak_ttc(
        agents_e, agents_c, cycle_sort_scheme=cycle_sort_scheme
    )
    weak_elapsed = perf_counter() - weak_start

    ttc_start = perf_counter()
    ttc_final_match_e, ttc_final_match_c = TTC(agents_e, agents_c)
    ttc_elapsed = perf_counter() - ttc_start

    react_start = perf_counter()
    react_final_match_e, react_final_match_c = ReactTTC_variant(agents_e, agents_c)
    react_elapsed = perf_counter() - react_start

    if DEBUG:
        debug_print_results(
            agents_e,
            agents_c,
            initial_endowment,
            weak_final_match_e,
            weak_final_match_c,
            ttc_final_match_e,
            ttc_final_match_c,
            react_final_match_e,
            react_final_match_c,
            preferences,
        )

    print(
        "WeakTTC total rank improvement:",
        total_rank_improvement(initial_endowment, weak_final_match_e, preferences),
    )
    print(f"WeakTTC runtime: {weak_elapsed:.6f} seconds")
    print(
        "TTC total rank improvement:",
        total_rank_improvement(initial_endowment, ttc_final_match_e, preferences),
    )
    print(f"TTC runtime: {ttc_elapsed:.6f} seconds")
    print(
        "ReACT-TTC total rank improvement:",
        total_rank_improvement(initial_endowment, react_final_match_e, preferences),
    )
    print(f"ReACT-TTC runtime: {react_elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
