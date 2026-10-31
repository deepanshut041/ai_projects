"""Standalone diagnostic figures from retained numerical review results."""

import os
from datetime import datetime


def make_review_plots(directory, rows, predictions, result, level_comparison):
    os.environ.setdefault("MPLCONFIGDIR", str(directory/".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    with plt.rc_context({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "bike-demand-review"}):
        def save(fig, name):
            for suffix in ("png", "svg"):
                path = directory/(name+"."+suffix)
                fig.savefig(path, bbox_inches="tight", metadata={"Creator": "Bike Demand Planner"})
                outputs.append(path.name)
            plt.close(fig)

        days = result["temporal"]["day"]
        times = [datetime.fromisoformat(d["start_date"]) for d in days]
        fig,axes = plt.subplots(2,1,figsize=(11,6),sharex=True,layout="constrained")
        axes[0].plot(times,[d["metrics"]["mean_signed_error"] for d in days],color="#2457a7")
        axes[0].axhline(0,color="#6b7280",linewidth=.8)
        axes[0].set(ylabel="Mean signed error\n(rentals/hour)",title="Group baseline: errors remain ordered in time")
        axes[1].plot(times,[d["metrics"]["mae"] for d in days],color="#a54b24")
        axes[1].set(ylabel="Daily MAE\n(rentals/hour)",xlabel="2012 validation date; daily means use actual observed counts")
        save(fig,"daily-errors")

        actual = np.asarray([r.target for r in rows]); values = np.asarray(predictions)
        upper = max(actual.max(),values.max())*1.03
        fig,ax = plt.subplots(figsize=(7,6),layout="constrained")
        ax.scatter(actual,values,s=8,alpha=.2,color="#2457a7",rasterized=True)
        ax.plot([0,upper],[0,upper],color="#a54b24",linewidth=1,label="Prediction equals observed count")
        ax.set(xlabel="Observed rentals/hour",ylabel="Predicted rentals/hour",xlim=(0,upper),ylim=(0,upper),
               title=f"Group baseline: {len(rows):,} unchanged validation hours")
        ax.legend(fontsize=8,loc="upper left")
        save(fig,"actual-versus-predicted")

        slices = result["slices"]
        fig,axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
        for ax,dimension,title in zip(axes.flat,["hour","workingday","month","season"],
                                      ["Hour of day","Working-day flag","Observed validation months","Seasonal coverage in this window"],strict=True):
            entries = [(k,v) for k,v in slices[dimension].items() if v["count"] or dimension=="season"]
            labels = [k for k,_ in entries]
            scores = [v["metrics"]["mae"] if v["metrics"] else float("nan") for _,v in entries]
            ax.bar(labels,scores,color="#2457a7")
            ax.set(title=title,ylabel="MAE (rentals/hour)")
            ax.set_xticks(range(len(labels)),labels)
            ax.set_xlim(-.6,len(labels)-.4)
            if dimension != "hour":
                for i,(_,entry) in enumerate(entries):
                    note = f'n={entry["count"]}' if entry["count"] else 'n=0; no score'
                    ax.annotate(note,(i,scores[i] if entry["count"] else 0),
                                ha="center",va="bottom",fontsize=8,xytext=(0,3),textcoords="offset points")
            ax.tick_params(axis="x",labelsize=8)
        fig.suptitle("Separate slice dimensions overlap; autumn has no validation observations",fontsize=12)
        save(fig,"calendar-slices")

        months = [str(m) for m in range(1,7)]; groups = level_comparison["groups"]
        x = np.arange(6)
        historical = [groups[m]["historical_cell_mean_reweighted"] if groups[m]["count"] else np.nan for m in months]
        observed = [groups[m]["validation_target_mean"] if groups[m]["count"] else np.nan for m in months]
        fig,ax = plt.subplots(figsize=(10,5.5),layout="constrained")
        ax.bar(x-.2,historical,width=.4,label="2011 calendar-cell means, weighted to 2012 membership",color="#2457a7")
        ax.bar(x+.2,observed,width=.4,label="Observed 2012 mean on shared cells",color="#a54b24")
        ax.set(xticks=x,xticklabels=["Jan","Feb","Mar","Apr","May","Jun"],ylabel="Mean rentals/hour",
               title="Matching month, hour and day type still leaves an observed level difference")
        ax.legend(fontsize=8,loc="upper left",bbox_to_anchor=(0,-.1))
        ax.text(.01,.98,"Retrospective diagnostic; not a new model or a causal estimate",transform=ax.transAxes,va="top",fontsize=9)
        save(fig,"calendar-levels")
    return {"matplotlib": matplotlib.__version__, "files": outputs, "scope": "descriptive validation figures; no test observations or uncertainty intervals"}
