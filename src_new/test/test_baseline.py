import os
import sys


CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, SRC_DIR)

from agents import initialize_agents, initialize_preferences, set_initial_endowment
from baseline import flatten_preferences


def build_example_instance():
    n_e = 5
    n_c = 10
    capacity = 2
    seed = 7

    agents_e, agents_c = initialize_agents(
        n_e=n_e, n_c=n_c, capacity=capacity, seed=seed
    )
    initial_endowment = set_initial_endowment(agents_e, agents_c, seed=seed)
    preferences = initialize_preferences(
        agents_e, agents_c, max_rank_size=4, output_path="/tmp/test_baseline_preferences.json", seed=seed
    )

    # Make the initial-assignment priority easy to inspect in printed output.
    agent_e = agents_e[0]
    initial_assignment = initial_endowment[agent_e.ID]
    tied_agent_c_ids = set(preferences[agent_e.ID][1])
    tied_agent_c_ids.add(initial_assignment)
    preferences[agent_e.ID][1] = tied_agent_c_ids
    agent_e.pref = preferences[agent_e.ID]

    return agents_e, agents_c, initial_endowment, preferences


def print_agents(agents_e, agents_c):
    print("AgentE:")
    for agent_e in agents_e:
        print(f"  {agent_e}")

    print("\nAgentC:")
    for agent_c in agents_c:
        print(f"  {agent_c}")


def print_flattened_preferences(initial_endowment, preferences):
    flattened_first = flatten_preferences(
        preferences, scheme="select_first", initial_match_e=initial_endowment
    )
    flattened_random = flatten_preferences(
        preferences, scheme="select_random", seed=11, initial_match_e=initial_endowment
    )

    print("\nInitial endowment:", initial_endowment)
    print("\nOriginal preferences:")
    for agent_e_id, pref in sorted(preferences.items()):
        print(f"  AgentE {agent_e_id}: {pref}")

    print("\nFlattened preferences with select_first:")
    for agent_e_id, pref in sorted(flattened_first.items()):
        print(f"  AgentE {agent_e_id}: {pref}")

    print("\nFlattened preferences with select_random:")
    for agent_e_id, pref in sorted(flattened_random.items()):
        print(f"  AgentE {agent_e_id}: {pref}")

    for agent_e_id, initial_assignment in initial_endowment.items():
        for rank, agent_c_ids in preferences[agent_e_id].items():
            if initial_assignment in agent_c_ids:
                assert flattened_first[agent_e_id][rank] == {initial_assignment}
                assert flattened_random[agent_e_id][rank] == {initial_assignment}


def main():
    agents_e, agents_c, initial_endowment, preferences = build_example_instance()
    print_agents(agents_e, agents_c)
    print_flattened_preferences(initial_endowment, preferences)


if __name__ == "__main__":
    main()
