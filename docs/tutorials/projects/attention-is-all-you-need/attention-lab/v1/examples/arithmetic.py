"""Independent scalar/list checks for the one-fixture attention teaching route."""
import json
import math

X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
V = [[2.0 * coordinate for coordinate in row] for row in X]
TARGET = [2.0, 0.0]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))

def retrieve(query, keys, values):
    scores = [dot(query, key) / math.sqrt(len(query)) for key in keys]
    exponentials = [math.exp(s - max(scores)) for s in scores]
    weights = [e / sum(exponentials) for e in exponentials]
    context = [sum(a * value[m] for a, value in zip(weights, values, strict=True))
               for m in range(len(values[0]))]
    return context, weights

def loss(query):
    context, _ = retrieve(query, X, V)
    return 0.5 * sum((c - t) ** 2 for c, t in zip(context, TARGET, strict=True))

query = X[0]
context, weights = retrieve(query, X, V)
error = [c - t for c, t in zip(context, TARGET, strict=True)]
score_gradient = [a * dot(error, [v - c for v, c in zip(value, context, strict=True)])
                  for a, value in zip(weights, V, strict=True)]
query_gradient = [sum(g * key[m] / math.sqrt(2)
                      for g, key in zip(score_gradient, X, strict=True)) for m in range(2)]
epsilon = 1e-6
for m in range(2):
    plus = query.copy(); minus = query.copy()
    plus[m] += epsilon; minus[m] -= epsilon
    finite_difference = (loss(plus) - loss(minus)) / (2 * epsilon)
    assert math.isclose(finite_difference, query_gradient[m], abs_tol=1e-8)
next_query = [q - 0.1 * g for q, g in zip(query, query_gradient, strict=True)]
assert loss(next_query) < loss(query)
assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(context, [1.3395230986533138, 1.0], strict=True))
supplied = [sum(a * value[m] for a, value in zip([0.4, 0.2, 0.3, 0.1], V, strict=True)) for m in range(2)]
print(json.dumps({'supplied_context': supplied,
                  'query_context': [round(x, 6) for x in context],
                  'query_gradient': [round(x, 6) for x in query_gradient],
                  'post_step_loss': round(loss(next_query), 6), 'passed': True}))
