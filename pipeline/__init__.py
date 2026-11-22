"""Course-catalog ingestion pipeline for the recommender/bandit thesis.

See docs/ingestion-plan.md. Three namespaces with a one-way dependency:

    catalog/  frozen facts from the sources          (read by everything)
    world/    oracle: skill DAG, user state, reward  (never read by the agent)
    agent/    arm index, feature matrices            (must not derive from world/)
"""
__all__ = ["paths", "hashing", "textnorm"]
