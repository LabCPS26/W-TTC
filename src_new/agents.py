import json
import os
import random


class AgentE:
    """
    @brief Represents an agent of type E.

    Each AgentE has a unique ID, one initial assignment to an AgentC,
    and a weak preference dictionary over AgentC IDs.
    """

    def __init__(self, agent_id):
        """
        @brief Construct an AgentE object.
        @param agent_id Unique integer identifier for the AgentE.
        """
        self.ID = agent_id
        self.initial_assignment = None
        self.pref = {}

    def __repr__(self):
        """
        @brief Return a readable string representation of the AgentE.
        @return String containing the AgentE state.
        """
        return f"AgentE(ID={self.ID}, initial_assignment={self.initial_assignment}, pref={self.pref})"


class AgentC:
    """
    @brief Represents an agent of type C.

    Each AgentC has a unique ID, a capacity, and the list of AgentE IDs
    assigned to it in the initial endowment.
    """

    def __init__(self, agent_id, capacity=1):
        """
        @brief Construct an AgentC object.
        @param agent_id Unique integer identifier for the AgentC.
        @param capacity Maximum number of AgentE objects assignable to this AgentC.
        """
        self.ID = agent_id
        self.capacity = capacity
        self.assigned = []

    def __repr__(self):
        """
        @brief Return a readable string representation of the AgentC.
        @return String containing the AgentC state.
        """
        return f"AgentC(ID={self.ID}, capacity={self.capacity}, assigned={self.assigned})"


def initialize_agents(n_e, n_c, capacity=1, capacity_type="strict", seed=None):
    """
    @brief Create the two agent sets E and C.
    @param n_e Number of AgentE objects to create.
    @param n_c Number of AgentC objects to create.
    @param capacity Maximum capacity assigned to each AgentC.
    @param capacity_type Capacity assignment mode: "strict" uses capacity for every
        AgentC; "loose" randomly assigns each AgentC capacity between 1 and capacity.
    @param seed Optional random seed for reproducible loose capacities.
    @return Tuple containing the AgentE list and AgentC list.
    @throws ValueError If capacity_type is unknown or capacity is less than 1.
    """
    if capacity < 1:
        raise ValueError("capacity must be at least 1.")
    if capacity_type not in {"strict", "loose"}:
        raise ValueError('capacity_type must be either "strict" or "loose".')

    agents_e = [AgentE(agent_id=i) for i in range(n_e)]
    agents_c = []
    rng = random.Random(seed)
    for i in range(n_c):
        agent_capacity = capacity
        if capacity_type == "loose":
            agent_capacity = rng.randint(1, capacity)
        agents_c.append(AgentC(agent_id=i, capacity=agent_capacity))
    return agents_e, agents_c


def set_initial_endowment(agents_e, agents_c, seed=None):
    """
    @brief Assign each AgentE to one AgentC while respecting AgentC capacities.
    @param agents_e List of AgentE objects.
    @param agents_c List of AgentC objects.
    @param seed Optional random seed for reproducible initial assignment.
    @return Dictionary mapping each AgentE ID to its assigned AgentC ID.
    @throws ValueError If total AgentC capacity is smaller than the number of AgentE objects.
    """
    available_agent_c = []
    for agent_c in agents_c:
        available_agent_c.extend([agent_c] * agent_c.capacity)

    if len(available_agent_c) < len(agents_e):
        raise ValueError("Total AgentC capacity is less than the number of AgentE agents.")

    rng = random.Random(seed)
    rng.shuffle(available_agent_c)

    for agent_e, agent_c in zip(agents_e, available_agent_c):
        agent_e.initial_assignment = agent_c.ID
        agent_c.assigned.append(agent_e.ID)

    return {agent_e.ID: agent_e.initial_assignment for agent_e in agents_e}


def initialize_preferences(agents_e, agents_c, max_rank_size=2, output_path=None, seed=None):
    """
    @brief Generate weak preferences for each AgentE over the AgentC set.
    @param agents_e List of AgentE objects whose preferences will be initialized.
    @param agents_c List of AgentC objects to rank.
    @param max_rank_size Maximum number of AgentC IDs that may share one rank.
    @param output_path Optional path where preferences will be written as JSON.
    @param seed Optional random seed for reproducible weak preferences.
    @return Dictionary mapping AgentE IDs to weak preference dictionaries.

    The in-memory preference format is:
    {agent_e_id: {rank: {agent_c_id, ...}}}

    The JSON file stores sets as sorted lists because JSON has no set type.
    """
    preferences = {}
    rng = random.Random(seed)

    for agent_e in agents_e:
        shuffled_agent_c_ids = [agent_c.ID for agent_c in agents_c]
        rng.shuffle(shuffled_agent_c_ids)

        rank = 1
        agent_e.pref = {}
        while shuffled_agent_c_ids:
            rank_size = rng.randint(1, max_rank_size)
            agent_e.pref[rank] = set(shuffled_agent_c_ids[:rank_size])
            shuffled_agent_c_ids = shuffled_agent_c_ids[rank_size:]
            rank = rank + 1

        preferences[agent_e.ID] = agent_e.pref

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "preferences.json")

    jsonable_preferences = {
        agent_e_id: {rank: sorted(agent_c_ids) for rank, agent_c_ids in pref.items()}
        for agent_e_id, pref in preferences.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jsonable_preferences, f, indent=2)

    return preferences
