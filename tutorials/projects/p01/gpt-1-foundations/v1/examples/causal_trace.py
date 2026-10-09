"""Illustrative zero-logit attention, not a learned model."""
import json

def attend(values, causal):
    return [round(sum(values[:i+1] if causal else values) / (i+1 if causal else len(values)), 9) for i in range(len(values))]

values=[2,4,8]
changed=[2,4,98]
tokens=[10,20,30,40]
print(json.dumps({"causal":attend(values, True),"unmasked":attend(values, False),"changed_prefix":attend(changed, True)[:2],"inputs":tokens[:-1],"targets":tokens[1:]}))
