# haskgrad

A reverse-mode automatic differentiation engine written in Haskell, inspired by [micrograd](https://github.com/karpathy/micrograd). Supports scalar and tensor operations with a neural network module, demonstrated by training an MLP to learn XOR.

## Quick start

```bash
cabal build
cabal run haskgrad
```

Change the seed in Main.hs to get different results.

```haskell
  rng <- mkRNG [SEED]
```

## Dependencies

- `base ^>=4.18.3.0`
- `containers`
