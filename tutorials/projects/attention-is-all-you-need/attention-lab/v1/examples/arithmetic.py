"""Independent standard-library checks for the Attention tutorial's arithmetic."""
import json
import math


def matmul(a, b):
    assert len(a[0]) == len(b)
    return [[sum(x*y for x,y in zip(row,col,strict=True)) for col in zip(*b)] for row in a]


def softmax(row):
    maximum = max(row)
    positive = [math.exp(x-maximum) for x in row]
    total = sum(positive)
    return [x/total for x in positive]


def near(actual, expected, tolerance=1e-10):
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for a,b in zip(actual,expected,strict=True): near(a,b,tolerance)
    else:
        assert math.isclose(actual,expected,rel_tol=tolerance,abs_tol=tolerance), (actual,expected)


def scalar_loss(q):
    weights = softmax([q,-q])
    context = weights[0]*2 + weights[1]*6
    return .5*(context-5)**2


def checks():
    values = [[2,4],[6,8]]
    near(matmul([[.5,.5]],values),[[4,6]])
    near(matmul([[.75,.25]],values),[[3,5]])
    near(matmul([[1,2,3],[4,5,6]],[[7,8],[9,10],[11,12]]),[[58,64],[139,154]])
    raw = softmax([1/math.sqrt(2),0])
    near(raw,[.6697615493266569,.3302384506733431])
    near(softmax([0,math.log(3)]),[.25,.75])
    near(softmax([1000,1000+math.log(3)]),[.25,.75])
    near(softmax([0,2*math.log(3)]),[.1,.9])
    x = [[1,0],[0,1],[1,1]]
    q = matmul(x,[[1,0],[1,1]])
    k = matmul(x,[[1,1],[0,-1]])
    v = [[a+.5,b-.5] for a,b in matmul(x,[[2,0],[0,3]])]
    near(q,[[1,0],[1,1],[2,1]])
    near(k,[[1,1],[0,-1],[1,0]])
    near(v,[[2.5,-.5],[.5,2.5],[2.5,2.5]])
    scores = matmul(q,list(map(list,zip(*k))))
    near(scores,[[1,0,1],[2,-1,1],[3,-1,2]])
    a = [softmax([s/math.sqrt(2) for s in row]) for row in scores]
    context = matmul(a,v)
    near(context[0],[2.1044483707191435,1.296663721960642])
    shifted = matmul(a,[[first+1,second-2] for first,second in v])
    near(shifted,[[first+1,second-2] for first,second in context])
    merged = [[1,0,2,1],[0,1,1,2]]
    near(matmul(merged,[[1,0,1,0],[0,1,0,1],[1,0,0,1],[0,1,1,0]]),[[3,1,2,2],[1,3,2,2]])
    epsilon = 1e-6
    finite_difference = (scalar_loss(epsilon)-scalar_loss(-epsilon))/(2*epsilon)
    near(finite_difference,2,1e-8)
    assert scalar_loss(-.2) < scalar_loss(0)
    return {"weighted_context":[4,6], "rectangular_product":[[58,64],[139,154]],
            "projected_first_context":[round(v,6) for v in context[0]],
            "query_gradient":round(finite_difference,6),"post_step_loss":round(scalar_loss(-.2),6),
            "parameter_count":4*(9**2+9),"passed":True}


if __name__ == '__main__':
    print(json.dumps(checks(),sort_keys=True))
