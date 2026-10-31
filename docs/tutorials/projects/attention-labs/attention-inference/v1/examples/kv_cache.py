"""CPU component reference: causal prefill/chunks, ownership and exact cache size."""
import json
import math
import torch


def causal_attention(q, k, v, offset=0):
    """q: [B,H,S,Dk]; k/v: [B,H,T,Dk/Dv]; q positions start at offset."""
    query_positions = offset + torch.arange(q.shape[-2], device=q.device)
    key_positions = torch.arange(k.shape[-2], device=k.device)
    visible = key_positions[None, :] <= query_positions[:, None]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return scores.masked_fill(~visible, -torch.inf).softmax(-1) @ v


class KVCache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.owner = None
        self.k = self.v = None

    @torch.no_grad()
    def append(self, q, k_new, v_new, *, owner, start):
        if any(t.ndim != 4 for t in [q, k_new, v_new]):
            raise ValueError('Expected [batch,head,token,feature] tensors')
        if q.shape != k_new.shape or q.shape[:3] != v_new.shape[:3]:
            raise ValueError('MHA query/key shapes and value leading axes must match')
        if q.shape[-2] == 0 or q.shape[-1] == 0 or v_new.shape[-1] == 0:
            raise ValueError('Empty chunks or features are not supported')
        tensors = [q, k_new, v_new]
        if self.k is not None:
            tensors += [self.k, self.v]
        if any(t.dtype != q.dtype or t.device != q.device for t in tensors):
            raise ValueError('Dtype/device changed')
        past = 0 if self.k is None else self.k.shape[-2]
        if start != past or (self.owner is not None and self.owner != owner):
            raise ValueError('Wrong absolute position or request owner')
        if self.k is not None:
            if self.k.shape[:2] + self.k.shape[-1:] != k_new.shape[:2] + k_new.shape[-1:]:
                raise ValueError('Key layout changed')
            if self.v.shape[:2] + self.v.shape[-1:] != v_new.shape[:2] + v_new.shape[-1:]:
                raise ValueError('Value layout changed')
        # Build candidates first; invalid input must not partly update retained state.
        k = k_new.clone() if self.k is None else torch.cat([self.k, k_new], dim=-2)
        v = v_new.clone() if self.v is None else torch.cat([self.v, v_new], dim=-2)
        result = causal_attention(q, k, v, offset=past)
        self.owner, self.k, self.v = owner, k, v
        return result

    def payload_bytes(self):
        return sum(t.numel() * t.element_size() for t in [self.k, self.v] if t is not None)


def run_checks():
    torch.manual_seed(17)
    q = torch.randn(2, 2, 6, 3, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(2, 2, 6, 2, dtype=torch.float64)
    expected = causal_attention(q, k, v)
    cache = KVCache()
    outputs = []
    for start, end in [(0, 3), (3, 5), (5, 6)]:
        outputs.append(cache.append(q[:, :, start:end], k[:, :, start:end],
                       v[:, :, start:end], owner='request-a', start=start))
    actual = torch.cat(outputs, dim=-2)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert cache.payload_bytes() == 2 * 2 * 6 * (3 + 2) * 8 == 960
    # Top-left masking of a non-square decode matrix incorrectly keeps only key 0.
    wrong = causal_attention(q[:, :, -1:], k, v, offset=0)
    assert not torch.allclose(wrong, expected[:, :, -1:])
    original = cache.k.clone()
    for owner, start in [('request-b', 6), ('request-a', 0)]:
        try:
            cache.append(q[:, :, :1], k[:, :, :1], v[:, :, :1], owner=owner, start=start)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid request/position accepted')
        assert torch.equal(cache.k, original)
    cache.reset()
    reused = cache.append(q, k, v, owner='request-b', start=0)
    torch.testing.assert_close(reused, expected)
    # Real zero-valued tokens stay valid keys. Padding needs separate validity metadata.
    z = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    values = torch.tensor([[[[2.0], [4.0]]]], dtype=torch.float64)
    assert causal_attention(z, z, values)[0, 0, -1, 0].item() == 3.0
    return {'chunks_equal_full': True, 'cache_bytes': 960,
            'wrong_offset_detected': True, 'owner_and_position_rejected': True,
            'reset_equal_full': True, 'zero_token_is_valid': True}


if __name__ == '__main__':
    print(json.dumps(run_checks(), sort_keys=True))
