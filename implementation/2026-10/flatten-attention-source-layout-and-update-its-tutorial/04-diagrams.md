# Organization and numerical visual preservation

The source tree shown in the README/setup lesson is a responsibility map, not a tensor execution diagram. core owns the reusable computation; data/training hold pending later stages; cli invokes it; utils draws it; notebooks imports and explains it. This distinction avoids implying that folder order is forward-pass order.

The attention flow diagrams, mathematical explanation and controlled visual fixtures retain their meaning. Moving the graph module changes import/output location, not its Q/K/V calculations. Full notebook executions regenerate 28 figures in the project runs directory. Fifteen selected figures are refreshed into the tutorial with captured drawing-source provenance. Browser checks inspect equations, figures, local scrolling, folder navigation and current source downloads.
