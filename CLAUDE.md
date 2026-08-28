# CLAUDE.md — id project

## Environment

**Use the `id` conda env, not `base`.** This overrides the global "use base" rule.

```bash
conda run -n id python run.py <stage>
```

DADApy (the GRIDE intrinsic-dimension estimator) pins `numpy<2`. Installing it into
`base` silently downgrades numpy there and affects every other project. `base` is
deliberately kept on numpy 2.x and must stay dadapy-free.

## Conventions

- Real runs happen on a remote H100, not this Mac. Keep device/dtype resolution
  automatic (`extract.resolve_device`); don't tune defaults down for local hardware.
- Validate pipeline changes locally with `--model hf-internal-testing/tiny-random-LlamaForCausalLM`
  on tiny `--n-*` values. Probes should land at chance — that is the label-leakage check.
- Prefer reference implementations of published methods (dadapy for GRIDE, SentEval's
  own data files) over reimplementations, so numbers stay comparable to the papers.
