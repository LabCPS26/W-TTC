import os
import sys


CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, SRC_DIR)

from agents import AgentC, AgentE
from weakTTC import run_weak_ttc


def make_agent_e(agent_id, initial_assignment, pref):
    agent_e = AgentE(agent_id)
    agent_e.initial_assignment = initial_assignment
    agent_e.pref = pref
    return agent_e


def make_agent_c(agent_id, capacity, assigned):
    agent_c = AgentC(agent_id, capacity=capacity)
    agent_c.assigned = list(assigned)
    return agent_c


def run_case(name, agents_e, agents_c):
    print(f"\n\n##### {name} #####")
    print("AgentE:")
    for agent_e in agents_e:
        print(f"  {agent_e}")

    print("AgentC:")
    for agent_c in agents_c:
        print(f"  {agent_c}")

    final_match_e, final_match_c = run_weak_ttc(agents_e, agents_c, debug=True, max_rounds=10)
    print("\nFinal AgentE matching:", final_match_e)
    print("Final AgentC matching:", final_match_c)


def real_cycle_instance():
    agents_e = [
        make_agent_e(0, initial_assignment=0, pref={1: {1}}),
        make_agent_e(1, initial_assignment=1, pref={1: {0}}),
    ]
    agents_c = [
        make_agent_c(0, capacity=1, assigned=[0]),
        make_agent_c(1, capacity=1, assigned=[1]),
    ]
    return agents_e, agents_c


def virtual_capacity_instance():
    agents_e = [
        make_agent_e(0, initial_assignment=0, pref={1: {1}}),
    ]
    agents_c = [
        make_agent_c(0, capacity=1, assigned=[0]),
        make_agent_c(1, capacity=2, assigned=[]),
    ]
    return agents_e, agents_c


def mixed_instance():
    agents_e = [
        make_agent_e(0, initial_assignment=0, pref={1: {1}}),
        make_agent_e(1, initial_assignment=1, pref={1: {2}}),
        make_agent_e(2, initial_assignment=2, pref={1: {0}}),
    ]
    agents_c = [
        make_agent_c(0, capacity=1, assigned=[0]),
        make_agent_c(1, capacity=2, assigned=[1]),
        make_agent_c(2, capacity=1, assigned=[2]),
    ]
    return agents_e, agents_c


def main():
    for name, instance_fn in [
        ("real cycle", real_cycle_instance),
        ("virtual capacity", virtual_capacity_instance),
        ("mixed real and virtual", mixed_instance),
    ]:
        run_case(name, *instance_fn())


if __name__ == "__main__":
    main()
