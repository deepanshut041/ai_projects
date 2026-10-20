"""Local assessment view beside the original target-free validation predictor."""

from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from .application import ReplayApplication, make_server
from .artifacts import digest, read_json


class FinalApplication(ReplayApplication):
    def __init__(self, bundle: Path, application_data: Path, assessment: Path, protocol: Path):
        super().__init__(bundle, application_data)
        self.assessment = read_json(assessment)
        self.protocol = read_json(protocol)
        if (self.assessment.get("version") != "hourly-final-assessment-v1" or self.assessment["status"] != "completed"
                or not self.assessment["test_evaluated"] or self.assessment["selection_changed"]
                or self.assessment["protocol_sha256"] != digest(protocol.read_bytes())
                or self.assessment["bundle_sha256"] != self.predictor.identity
                or self.assessment["selected"] != "groups"
                or self.assessment["results"]["groups"]["model_sha256"] != self.predictor.candidate["model_sha256"]):
            raise ValueError("Assessment, protocol and saved predictor identities differ.")
        for name, identity in self.assessment["artifacts_sha256"].items():
            if Path(name).name != name or digest((assessment.parent/name).read_bytes()) != identity:
                raise ValueError("Assessment evidence changed.")
        for name, result in self.assessment["results"].items():
            if any(result[key] != value for key,value in self.protocol["choices"][name].items()):
                raise ValueError("Assessment choice differs from the frozen protocol.")

    def info(self):
        return {**super().info(), "final_test_evaluated":True,
                "assessment_protocol_sha256":self.assessment["protocol_sha256"],
                "scope":"fixed final assessment; interactive requests remain historical validation replay"}

    def summary(self):
        r=self.assessment
        return {"selected":r["selected"],"protocol_sha256":r["protocol_sha256"],"coverage":r["coverage"],
                "bundle_sha256":r["bundle_sha256"],"examples":r["examples"],"source_provenance":r["source_provenance"],
                "cost_scope":r["cost_scope"],"interpretation":r["interpretation"],
                "rows":[{"name":name,"policy":v["policy"],"selected":name==r["selected"],
                         "test":v["test"]["metrics"],"validation":v["validation"],
                         "workingday_rush":v["slices"]["workingday_rush"]["yes"]["metrics"],
                         "months":v["slices"]["month"],"fit":v["fit"],"model_json_bytes":v["model_json_bytes"],
                         "predict_batch_seconds":v["predict_batch_seconds"],"model_sha256":v["model_sha256"]}
                        for name,v in r["results"].items()]}


def make_final_server(application: FinalApplication, port: int = 8504):
    server=make_server(application,port)
    base=server.RequestHandlerClass
    class Handler(base):
        def do_GET(self):
            path=urlsplit(self.path).path
            if path=="/api/assessment":
                return self.send(200,application.summary())
            if path in ("/","/app.js","/style.css"):
                name={"/":"final.html","/app.js":"final.js","/style.css":"final.css"}[path]
                kind={"/":"text/html","/app.js":"text/javascript","/style.css":"text/css"}[path]
                return self.send(200,files("bike_demand_planner").joinpath("web",name).read_bytes(),kind+"; charset=utf-8")
            return super().do_GET()
    server.RequestHandlerClass=Handler
    return server
