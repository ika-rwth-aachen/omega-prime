@dataclass
class MetricManager:
    metrics: list[Metric] = field(default_factory=lambda: metrics)
    """List of metrics to compute"""
    exclude_columns: list[str] = field(default_factory=list)
    """List of columns computed by the metrics that do not need to be computed"""
    exclude_properties: list[str] = field(default_factory=list)
    """List of tables in the properties dict that do not need to be computed"""
    _dependencies: dict[int | str, list[int | str]] = field(init=False)
    """Automatically derived dependencies between metrics"""
    _ordered_metrics: list[Metric] = field(init=False)
    """Automatically derived execution order of metrics"""
    _parameters: list = field(init=False)
    """Automatically derived list of parameters to keep"""

    def __post_init__(self):
        self._dependencies = {
            val: [i]
            for i, m in enumerate(self.metrics)
            for val in [f"column_{n}" for n in m.computes_columns + m.computes_intermediate_columns]
            + [f"property_{n}" for n in m.computes_properties + m.computes_intermediate_properties]
        } | {
            i: [f"column_{n}" for n in m.requires_columns] + [f"property_{n}" for n in m.requires_properties]
            for i, m in enumerate(self.metrics)
        }

        unresovled_dependencies = {
            k: v for k, vv in self._dependencies.items() for v in vv if v not in self._dependencies
        }
        if len(unresovled_dependencies) > 0:
            error_dict = {f"self.metrics[{k}]": v for k, v in unresovled_dependencies.items()}
            raise RuntimeError(
                f"There are columns and properties required by metrics, that are never computed: {error_dict}"
            )

        self._parameters = [v for m in self.metrics for v in m._parameters]

        self.exclude_columns += [v for m in self.metrics for v in m.computes_intermediate_columns]
        self.exclude_properties += [v for m in self.metrics for v in m.computes_intermediate_properties]

        ts = graphlib.TopologicalSorter(self._dependencies)
        self._ordered_metrics = [self.metrics[o] for o in ts.static_order() if isinstance(o, int)]

    def __repr__(self):
        return f"computes columns: {[c for m in self._ordered_metrics for c in m.computes_columns]} - computes properties {[p for m in self._ordered_metrics for p in m.computes_properties]} - parameters {list(set([str(m) for m in self._parameters]))}"

    def compute(self, r: Recording, **kwargs) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
        if "polygon" not in r._df.columns:
            r._df = r._add_polygons(r._df)
        if "geometry" not in r._df.columns:
            r._df = r._df.with_columns(geometry=st.from_shapely("polygon"))

        df = pl.LazyFrame(r._df)
        properties = {}
        for m in self._ordered_metrics:
            df, new_p = m.compute_lazy(
                df=df,
                **{k: properties[k] for k in m.requires_properties},
                **{k: v for k, v in kwargs.items() if k in [p.name for p in m._parameters]},
            )
            properties |= new_p
        for k in self.exclude_properties:
            del properties[k]
        df = df.drop(self.exclude_columns)
        res = pl.collect_all([df] + list(properties.values()))
        df, computed_props = res[0], res[1:]
        assert all(c in df.columns or c in self.exclude_columns for m in self.metrics for c in m.computes_columns)
        return df, {k: v for k, v in zip(properties.keys(), computed_props)}

    def plot_dependencies(self):
        import networkx as nx
        import matplotlib.pyplot as plt

        i = 0
        pos = {}
        G = nx.DiGraph()

        for m in self._ordered_metrics:
            n = m.compute_func.__name__
            pos[n] = [0, -i]
            i += 1
            cn = [f"column_{c}" for c in m.computes_columns + m.computes_intermediate_columns] + [
                f"property_{c}" for c in m.computes_properties + m.computes_intermediate_properties
            ]
            pos |= {k: [1 + j, -i] for j, k in enumerate(cn)}
            G.add_node(n, color="lightblue")
            for c in cn:
                G.add_node(c, color="lightgreen")
                G.add_edge(n, c, label="computes")
            for r in [f"column_{c}" for c in m.requires_columns] + [f"property_{p}" for p in m.requires_properties]:
                G.add_edge(r, n, label="required by")
            i += 1

        # Draw nodes and edges
        fig, ax = plt.subplots()
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=2000,
            node_color=list(nx.get_node_attributes(G, "color").values()),
            arrows=True,
            font_size=8,
            ax=ax,
        )

        return fig
