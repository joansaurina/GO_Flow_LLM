import networkx as nx
import matplotlib.pyplot as plt
import polars as pl
import click

import holoviews as hv

hv.extension("bokeh")
from bokeh.plotting import show

from mirna_curator.flowchart import curation


# I hate this but it works for now
def get_edges_count(recorded_df: pl.DataFrame, cf: curation.CurationFlowchart):

    cf_true_edges = []
    cf_false_edges = []
    for node in cf.nodes.items():
        if str(node[1].type).startswith("terminal"):
            continue
        start_node = node[0]
        cf_true_edges.append((start_node, node[1].transitions.true))
        cf_false_edges.append((start_node, node[1].transitions.false))

    decision_nodes = [
        name for name, node in cf.nodes.items() if not str(node.type).startswith("terminal")
    ]

    # Calculate counts for all decision nodes at once
    stats = recorded_df.select(
        [pl.col(f"{node}_result").is_not_null().sum().alias(f"{node}_total") for node in decision_nodes] +
        [pl.col(f"{node}_result").sum().alias(f"{node}_true") for node in decision_nodes]
    )
    stats_dict = stats.to_dicts()[0]

    edges_count = {}
    for edge in cf_true_edges:
        node = edge[0]
        edges_count[edge] = stats_dict.get(f"{node}_true", 0)
    for edge in cf_false_edges:
        node = edge[0]
        edges_count[edge] = stats_dict.get(f"{node}_total", 0) - stats_dict.get(f"{node}_true", 0)

    return edges_count


def create_miRNA_flowchart_viz(
    recorded_df: pl.DataFrame,
    expected_df: pl.DataFrame,
    filter_class: int,
    cf: curation.CurationFlowchart,
):
    """
    Create a visualization of the miRNA flowchart with node statistics.

    Parameters:
    recorded_df: Polars DataFrame with columns for each node and the actual outcomes
    expected_df: Polars DataFrame with columns for each node and the expected outcomes
    """
    # Create directed graph
    G = nx.DiGraph()

    # Define nodes and their positions (manually set to match original layout)
    pos = {
        "experimental_evidence": (0, 4),
        "functional_interaction": (0, 3),
        "effect_endogenous_1": (0, 2),
        "mirna_mrna_binding": (2, 3),
        "effect_endogenous_2": (2, 2),
        "mirna_changes": (2, 1),
        "computational_prediction": (4, 3),
        "effect_endogenous_3": (6, 2),
        "no_annotation": (6, 3),
        "validated_binding_only": (0, 0),
        "validated_binding_mrna": (2, 0),
        "validated_binding_translation": (4, 0),
        "no_validated_binding": (6, 0),
    }

    # Generate node labels dynamically from flowchart
    node_labels = {}
    for name, node in cf.nodes.items():
        node_labels[name] = node.data.desc if node.data.desc else name

    edges = []
    edge_colours = []
    for node in cf.nodes.items():
        if str(node[1].type).startswith("terminal"):
            continue
        start_node = node[0]
        edges.append((start_node, node[1].transitions.true))
        edge_colours.append("g")
        edges.append((start_node, node[1].transitions.false))
        edge_colours.append("r")

    G.add_edges_from(edges)

    node_columns = list(cf.nodes.keys())

    # Filter out terminal nodes for the result casting
    decision_nodes = [
        name for name, node in cf.nodes.items() if not str(node.type).startswith("terminal")
    ]

    recorded_df = recorded_df.with_columns(
        [pl.col(col).cast(pl.Boolean) for col in node_columns]
    )
    recorded_df = recorded_df.with_columns(
        [pl.col(f"{col}_result").cast(pl.Boolean) for col in decision_nodes]
    )

    expected_df = expected_df.with_columns(
        [pl.col(col).cast(pl.Boolean) for col in node_columns]
    )
    expected_df = expected_df.with_columns(
        [pl.col(f"{col}_result").cast(pl.Boolean) for col in decision_nodes]
    )

    expected_columns = expected_df.columns
    expected_to_count = (
        expected_df.join(recorded_df, on="PMCID", how="inner")
        .select(expected_columns)
        .unique("PMCID")
    )

    expected_node_count = get_edges_count(expected_to_count, cf)
    recorded_columns = recorded_df.columns
    recorded_to_count = (
        recorded_df.join(expected_df, on="PMCID", how="inner")
        .select(recorded_columns)
        .unique("PMCID")
    )
    recorded_node_count = get_edges_count(recorded_to_count, cf)

    edges_labels = {}
    for edge in edges:
        recorded_edge_count = recorded_node_count[edge]
        expected_edge_count = expected_node_count[edge]
        edges_labels[edge] = (
            f"Recorded: {recorded_edge_count}\nExpected: {expected_edge_count}"
        )

    plt.figure(figsize=(15, 10))
    plt.title(f"Flowchart results for class {filter_class}", y=1.05, color="k")

    nx.draw(G, pos, node_size=1000)
    for node, (x, y) in pos.items():
        plt.text(
            x,
            y,
            node_labels.get(node, node),
            horizontalalignment="center",
            wrap=True,
            bbox=dict(facecolor="white", alpha=0.5, boxstyle="round"),
        )
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colours, arrows=True, arrowsize=20)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_labels)

    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    if filter_class is not None:
        plt.savefig(f"flowchart_class_{filter_class}.png")
    else:
        plt.title("Flowchart results for all classes", pad=20, y=1.02)
        plt.savefig("flowchart_class_all.png")

    return plt.gcf()


def create_sankey_df(raw_df, cf: curation.CurationFlowchart):
    # Generate lookup dynamically from flowchart descriptions
    node_label_lookup = {name: node.data.desc for name, node in cf.nodes.items() if node.data.desc}

    sankey_data = []
    ## figure out edges in the curation flowchart

    for node_name, node in cf.nodes.items():
        if node.type == "decision":

            check = raw_df.filter(
                pl.col(f"{node_name}_result").is_not_null()
            ).is_empty()
            if check:
                true_count = 0
                false_count = 0
                continue
            else:
                true_count = (
                    raw_df.filter(pl.col(f"{node_name}_result").is_not_null())
                    .select(pl.col(f"{node_name}_result").sum())
                    .to_numpy()
                    .flatten()[0]
                )
                if true_count > 0:
                    sankey_data.append(
                        {
                            "source": node_name,
                            "target": node.transitions.true,
                            "value": true_count,
                            "type": "yes",
                        }
                    )

                false_count = (
                    raw_df.filter(pl.col(f"{node_name}_result").is_not_null())
                    .select(pl.col(f"{node_name}_result").not_().sum())
                    .to_numpy()
                    .flatten()[0]
                )
                if false_count > 0:
                    sankey_data.append(
                        {
                            "source": node_name,
                            "target": node.transitions.false,
                            "value": false_count,
                            "type": "no",
                        }
                    )
    # Modify the source and target names using the mapping
    modified_sankey_data = []
    for row in sankey_data:
        modified_row = row.copy()  # Create a copy to avoid modifying original data
        modified_row["source"] = node_label_lookup.get(row["source"], row["source"])
        modified_row["target"] = node_label_lookup.get(row["target"], row["target"])
        modified_row["node_color"] = "#3498db"
        modified_sankey_data.append(modified_row)

    sankey_df = pl.DataFrame(modified_sankey_data)
    return sankey_df


@click.command()
@click.option("--recorded_df", help="Path to the recorded data file")
@click.option("--expected_df", help="Path to the expected data file")
@click.option(
    "--filter_class",
    default=None,
    help="Filter to one class, None means no filtering",
    type=int,
)
@click.option("--flowchart", help="Path to the flowchart JSON file", required=True)
def main(recorded_df, expected_df, filter_class, flowchart):
    cf = curation.CurationFlowchart.model_validate_json(
        open(flowchart, "r").read()
    )

    recorded_df = pl.read_parquet(recorded_df).unnest("curation_result")
    expected_df = pl.read_parquet(expected_df)

    sankey_df_recorded = create_sankey_df(recorded_df, cf)

    color_map = {"yes": "g", "no": "r"}
    edge_colors = sankey_df_recorded.get_column("type").to_list()
    edge_colors = [color_map[t] for t in edge_colors]

    edges = list(
        zip(
            sankey_df_recorded.get_column("source").to_list(),
            sankey_df_recorded.get_column("target").to_list(),
            sankey_df_recorded.get_column("value").to_list(),
            sankey_df_recorded.get_column("type").to_list(),
            sankey_df_recorded.get_column("node_color").to_list(),
        )
    )

    # Get unique node names from both start and end points
    nodes = set([edge[0] for edge in edges] + [edge[1] for edge in edges])

    sankey = hv.Sankey(
        edges, kdims=["start", "end"], vdims=["Value", "type", "node_color"]
    )
    sankey.opts(
        edge_color="type",
        edge_cmap={"yes": "#2ecc71", "no": "#e74c3c"},
        node_cmap="node_color",
        edge_alpha=0.6,
        edge_line_width=2,
        width=2400 // 2,
        height=1600 // 2,
        label_position="right",
    )
    show(hv.render(sankey))


if __name__ == "__main__":
    main()
