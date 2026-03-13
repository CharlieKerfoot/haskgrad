module Autograd
  ( Value (..),
    val,
    getData,
    setData,
    getGrad,
    setGrad,
    zeroGrad,
    addGrad,
    backward,
    add,
    mul,
    sub,
    neg,
    divide,
    power,
    expV,
    logV,
    relu,
    sigmoid,
    tanhV,
    softmax,
    foldl1M,
  )
where

import Data.IORef
import qualified Data.Set as S
import System.IO.Unsafe (unsafePerformIO)

-- | Global UID counter for unique identification of Values
{-# NOINLINE uidCounter #-}
uidCounter :: IORef Int
uidCounter = unsafePerformIO $ newIORef 0

freshUID :: IO Int
freshUID = atomicModifyIORef' uidCounter (\n -> (n + 1, n))

-- | Core autograd value with mutable data, grad, and backward closure
data Value = Value
  { _data :: !(IORef Double),
    _grad :: !(IORef Double),
    _backward :: IO (),
    _children :: [Value],
    _uid :: !Int
  }

instance Eq Value where
  a == b = _uid a == _uid b

instance Ord Value where
  compare a b = compare (_uid a) (_uid b)

instance Show Value where
  show v = "Value(uid=" ++ show (_uid v) ++ ")"

-- | Smart constructor for leaf values
val :: Double -> IO Value
val x = Value <$> newIORef x <*> newIORef 0.0 <*> pure (return ()) <*> pure [] <*> freshUID

-- | Accessors
getData :: Value -> IO Double
getData = readIORef . _data

setData :: Value -> Double -> IO ()
setData = writeIORef . _data

getGrad :: Value -> IO Double
getGrad = readIORef . _grad

setGrad :: Value -> Double -> IO ()
setGrad = writeIORef . _grad

zeroGrad :: Value -> IO ()
zeroGrad = flip writeIORef 0.0 . _grad

addGrad :: Value -> Double -> IO ()
addGrad v = modifyIORef' (_grad v) . (+)

-- | Binary operation combinator
-- gradFn takes (aData, bData, outGrad) and returns (gradA, gradB)
binOp ::
  (Double -> Double -> Double) ->
  (Double -> Double -> Double -> (Double, Double)) ->
  Value ->
  Value ->
  IO Value
binOp f gradFn a b = do
  da <- getData a
  db <- getData b
  d <- newIORef (f da db)
  g <- newIORef 0.0
  u <- freshUID
  let bwd = do
        og <- readIORef g
        let (ga, gb) = gradFn da db og
        addGrad a ga
        addGrad b gb
  return $ Value d g bwd [a, b] u

-- | Unary operation combinator
unaryOp ::
  (Double -> Double) ->
  (Double -> Double -> Double) ->
  Value ->
  IO Value
unaryOp f gradFn a = do
  da <- getData a
  d <- newIORef (f da)
  g <- newIORef 0.0
  u <- freshUID
  let bwd = do
        og <- readIORef g
        addGrad a (gradFn da og)
  return $ Value d g bwd [a] u

-- | Arithmetic operations
add :: Value -> Value -> IO Value
add = binOp (+) (\_ _ og -> (og, og))

mul :: Value -> Value -> IO Value
mul = binOp (*) (\da db og -> (og * db, og * da))

sub :: Value -> Value -> IO Value
sub = binOp (-) (\_ _ og -> (og, negate og))

divide :: Value -> Value -> IO Value
divide = binOp (/) (\_ db og -> (og / db, negate og / (db * db)))

-- | Unary operations
neg :: Value -> IO Value
neg = unaryOp negate (const negate)

expV :: Value -> IO Value
expV = unaryOp exp (\da og -> og * exp da)

logV :: Value -> IO Value
logV = unaryOp log (flip (/))

power :: Value -> Double -> IO Value
power a e = unaryOp (** e) (\da og -> og * e * (da ** (e - 1))) a

-- | Activation functions
relu :: Value -> IO Value
relu = unaryOp (max 0) (\da og -> if da > 0 then og else 0)

sigmoid :: Value -> IO Value
sigmoid = unaryOp sig (\da og -> let s = sig da in og * s * (1 - s))
  where
    sig x = 1 / (1 + exp (negate x))

tanhV :: Value -> IO Value
tanhV = unaryOp tanh (\da og -> og * (1 - tanh da ^ (2 :: Int)))

-- | Softmax over a list of Values
-- Forward: si = exp(xi - max) / sum(exp)
-- Backward per output i: dL/dxj += dL/dsi * si * (delta_ij - sj)
softmax :: [Value] -> IO [Value]
softmax xs = do
  ds <- mapM getData xs
  let maxD = maximum ds
      exps_ = map (\d -> exp (d - maxD)) ds
      sumExps = sum exps_
      sVals = map (/ sumExps) exps_
  -- Create output nodes
  outNodes <-
    mapM
      ( \sv ->
          Value <$> newIORef sv <*> newIORef 0.0 <*> pure (return ()) <*> pure [] <*> freshUID
      )
      sVals
  -- Wire backward: each output i knows its index and all softmax values
  let wired =
        zipWith3
          ( \i si outV ->
              outV
                { _backward = do
                    og <- readIORef (_grad outV)
                    mapM_
                      ( \(j, sj, xj) ->
                          let delta = if i == j then 1.0 else 0.0
                           in addGrad xj (og * si * (delta - sj))
                      )
                      (zip3 [(0 :: Int) ..] sVals xs),
                  _children = xs
                }
          )
          [(0 :: Int) ..]
          sVals
          outNodes
  return wired

-- | Topological sort using Data.Set for visited tracking
topoSort :: Value -> [Value]
topoSort root = snd $ go S.empty [] root
  where
    go visited order v
      | S.member (_uid v) visited = (visited, order)
      | otherwise =
          let (visited', order') =
                foldl
                  (\(vis, ord) c -> go vis ord c)
                  (S.insert (_uid v) visited, order)
                  (_children v)
           in (visited', v : order')

-- | Reverse-mode autodiff via topological sort
backward :: Value -> IO ()
backward root = do
  writeIORef (_grad root) 1.0
  mapM_ _backward (topoSort root)

-- | Fold over a non-empty list
foldl1M :: (Monad m) => (a -> a -> m a) -> [a] -> m a
foldl1M _ [] = error "foldl1M: empty list"
foldl1M _ [x] = return x
foldl1M f (x : xs) = go x xs
  where
    go acc [] = return acc
    go acc (y : ys) = f acc y >>= \acc' -> go acc' ys
