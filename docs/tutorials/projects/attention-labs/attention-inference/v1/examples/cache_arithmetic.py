import json
import math


def softmax(row):
    values = [math.exp(x-max(row)) for x in row]
    return [x/sum(values) for x in values]


def run():
    q = [[.2,.3,.4,.1], [.4,.1,.2,.3], [.5,.5,.3,.4], [.3,.4,.1,.2]]
    k = [[.1,.2,.5,.2], [.3,.4,.2,.1], [.5,.3,.1,.4], [.2,.1,.3,.3]]
    scores = [[sum(a*b for a,b in zip(qi,kj))/2 for kj in k] for qi in q]
    assert abs(scores[0][0]-.15) < 1e-15
    q_new = [.2,.6,.1,.4]
    keys = k + [[.7,.2,.3,.1]]
    weights = softmax([sum(a*b for a,b in zip(q_new,kj))/2 for kj in keys])
    values = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,1,1]]
    context = [sum(w*v[j] for w,v in zip(weights,values)) for j in range(4)]
    assert abs(sum(weights)-1) < 1e-15
    memory = 2*1*4096*8*128*80*2
    return {'scores': [[round(x,6) for x in row] for row in scores],
            'new_weights': [round(x,6) for x in weights],
            'new_context': [round(x,6) for x in context],
            'gqa_cache_bytes': memory, 'gqa_cache_gib': memory/2**30,
            'mha_cache_gib': memory*8/2**30}


if __name__ == '__main__':
    print(json.dumps(run(),sort_keys=True))
