"""
generate_pm_dataset.py
======================

This script generates a synthetic dataset representing end‑to‑end warehouse automation
projects for GreyOrange. Each row models a deployment project from the perspective of a
project manager overseeing installation of warehouse robotics. The goal is to capture
realistic variability in project attributes such as warehouse size, robot mix, team size,
risk level, tool utilisation and resulting schedule and cost outcomes.

The synthetic dataset is designed to support analysis of key business questions such as:

* How does project complexity (warehouse size and number of robots) influence schedule and cost variance?
* Does the mix of HTM (horizontal tote movers) and VTM (vertical tote movers) affect project outcomes?
* Are deployment planning tools (VOS, CAD, Tower) associated with shorter commissioning times and fewer overruns?
* How do risk categories correlate with deviations from estimated duration and cost?
* Do projects in certain regions or with particular team sizes tend to perform better?

Running this script will create a CSV file (`pm_project_dataset.csv`) containing the
synthetic data. The number of projects can be customised via the `--n_projects` argument.

Example usage:

```
python generate_pm_dataset.py --output pm_project_dataset.csv --n_projects 300 --seed 42
```

The script uses a deterministic random seed for reproducibility. The distribution
assumptions are deliberately broad to capture diverse project scenarios while still
reflecting typical warehouse automation deployments.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def generate_projects(n_projects: int, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic dataset of warehouse automation projects.

    Parameters
    ----------
    n_projects : int
        Number of project records to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing synthetic project data.
    """
    rng = np.random.default_rng(seed)

    # Define categorical options
    regions = [
        "North America",
        "Europe",
        "Asia Pacific",
        "Latin America",
    ]
    risk_categories = ["Low", "Medium", "High"]

    project_ids: List[str] = [f"PRJ{str(i + 1).zfill(4)}" for i in range(n_projects)]
    region_choices = rng.choice(regions, size=n_projects)
    risk_choices = rng.choice(risk_categories, size=n_projects, p=[0.5, 0.35, 0.15])

    # Warehouse size in square feet: typical ranges from 50k to 500k
    warehouse_size_sqft = rng.uniform(50_000, 500_000, size=n_projects)

    # Number of robots: distribution depends on warehouse size
    # For each project, approximate total robots as roughly one robot per 1,500–3,000 sqft
    base_robot_rate = rng.uniform(1500, 3000, size=n_projects)
    total_robots = (warehouse_size_sqft / base_robot_rate).astype(int)

    # Split between HTMs and VTMs. VTMs typically used for vertical tote handling,
    # making up a smaller proportion of the fleet.
    vtm_ratio = rng.uniform(0.2, 0.5, size=n_projects)
    num_vtm = (total_robots * vtm_ratio).astype(int)
    num_htm = total_robots - num_vtm

    # Team size: correlate loosely with project scale (more robots & larger warehouses
    # require larger teams).
    team_size = (total_robots / 10 + rng.normal(15, 3, size=n_projects)).clip(min=5)
    team_size = team_size.astype(int)

    # Complexity index: derived from warehouse size and robot counts.
    # Normalised to a 0–1 range and mapped to categories.
    complexity_index = (
        (warehouse_size_sqft / warehouse_size_sqft.max()) * 0.6
        + (total_robots / total_robots.max()) * 0.4
    )
    complexity_category = pd.cut(
        complexity_index,
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Estimated duration in days: base of 60 days plus contribution from
    # complexity and risk. Higher complexity and risk increase duration.
    base_duration = 60
    duration_multiplier = 1 + complexity_index + rng.normal(0, 0.1, size=n_projects)
    risk_factor = risk_choices
    risk_multiplier = np.array(
        [0.8 if r == "Low" else 1.0 if r == "Medium" else 1.2 for r in risk_factor]
    )
    estimated_duration = base_duration * duration_multiplier * risk_multiplier
    estimated_duration = estimated_duration.clip(min=30, max=365)

    # Actual duration deviates from estimate depending on complexity and risk
    schedule_noise = rng.normal(0, 10, size=n_projects)
    actual_duration = estimated_duration + (complexity_index * 20) + schedule_noise
    actual_duration = actual_duration.clip(min=30)

    # Estimated cost in thousands USD: base cost plus contributions from size and robots
    base_cost = 500  # baseline cost (kUSD)
    size_cost_factor = warehouse_size_sqft / 10_000
    robot_cost_factor = total_robots * 5
    estimated_cost = base_cost + size_cost_factor + robot_cost_factor
    # Increase cost for high risk projects
    estimated_cost *= risk_multiplier

    # Actual cost deviates based on complexity and risk
    cost_noise = rng.normal(0, 100, size=n_projects)
    actual_cost = estimated_cost + (complexity_index * 200) + cost_noise
    actual_cost = actual_cost.clip(min=100)

    # Tool usage probabilities: assuming VOS and CAD are used by larger projects,
    # whereas Tower is used less frequently.
    uses_vos = rng.random(n_projects) < (
        0.3 + 0.5 * (complexity_index > 0.5).astype(float)
    )
    uses_cad = rng.random(n_projects) < (
        0.4 + 0.4 * (complexity_index > 0.4).astype(float)
    )
    uses_tower = rng.random(n_projects) < 0.25

    # Commissioning days: influenced by number of robots and tool usage. Tool usage is
    # expected to shorten commissioning.
    base_commissioning = (total_robots / 10) + 5
    tool_reduction = (
        uses_vos.astype(float) * 3 + uses_cad.astype(float) * 2 + uses_tower.astype(float) * 1
    )
    commissioning_days = base_commissioning - tool_reduction + rng.normal(0, 2, size=n_projects)
    commissioning_days = commissioning_days.clip(min=3)

    # Derived variances
    schedule_variance = actual_duration - estimated_duration
    cost_variance = actual_cost - estimated_cost

    df = pd.DataFrame(
        {
            "project_id": project_ids,
            "region": region_choices,
            "warehouse_size_sqft": warehouse_size_sqft.astype(int),
            "num_htm": num_htm,
            "num_vtm": num_vtm,
            "total_robots": total_robots,
            "team_size": team_size,
            "complexity_category": complexity_category.astype(str),
            "risk_category": risk_choices,
            "estimated_duration_days": estimated_duration.round(1),
            "actual_duration_days": actual_duration.round(1),
            "estimated_cost_kusd": estimated_cost.round(2),
            "actual_cost_kusd": actual_cost.round(2),
            "uses_vos": uses_vos,
            "uses_cad": uses_cad,
            "uses_tower": uses_tower,
            "commissioning_days": commissioning_days.round(1),
            "schedule_variance_days": schedule_variance.round(1),
            "cost_variance_kusd": cost_variance.round(2),
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic project management dataset for GreyOrange warehouse automation projects."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pm_project_dataset.csv",
        help="Output CSV filename (default: pm_project_dataset.csv)",
    )
    parser.add_argument(
        "--n_projects",
        type=int,
        default=300,
        help="Number of projects to generate (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    df = generate_projects(n_projects=args.n_projects, seed=args.seed)
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"Generated dataset with {len(df)} projects written to {output_path}")


if __name__ == "__main__":
    main()