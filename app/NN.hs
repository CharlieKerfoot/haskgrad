module NN
  ( RNG,
    mkRNG,
    nextRandom,
    Linear (..),
    MLP (..),
    newLinear,
    newMLP,
    forwardLinear,
    forwardMLP,
    parameters,
    zeroGrads,
    sgdStep,
  )
where

import Autograd
  ( Value,
    add,
    foldl1M,
    getData,
    getGrad,
    mul,
    relu,
    setData,
    sigmoid,
    val,
    zeroGrad,
  )
import Control.Monad (zipWithM)
import Data.IORef

newtype RNG = RNG (IORef Int)

mkRNG :: Int -> IO RNG
mkRNG seed = RNG <$> newIORef seed

nextRandom :: RNG -> IO Double
nextRandom (RNG ref) = do
  s <- readIORef ref
  let s' = (s * 1103515245 + 12345) `mod` (2 ^ (31 :: Int))
  writeIORef ref s'
  return $ fromIntegral s' / fromIntegral (2 ^ (30 :: Int) :: Int) - 1.0

data Linear = Linear
  { linWeights :: [Value],
    linBias :: [Value],
    linNIn :: !Int,
    linNOut :: !Int
  }

newtype MLP = MLP {mlpLayers :: [Linear]}

newLinear :: RNG -> Int -> Int -> IO Linear
newLinear rng nIn nOut = do
  let scale = sqrt (2.0 / fromIntegral (nIn + nOut))
  ws <- sequence [do r <- nextRandom rng; val (r * scale) | _ <- [1 .. nIn * nOut]]
  bs <- sequence [val 0.0 | _ <- [1 .. nOut]]
  return $ Linear ws bs nIn nOut

newMLP :: RNG -> [Int] -> IO MLP
newMLP rng sizes = do
  let pairs = zip sizes (tail sizes)
  ls <- mapM (uncurry (newLinear rng)) pairs
  return $ MLP ls

forwardLinear :: Linear -> [Value] -> IO [Value]
forwardLinear lin xs = do
  let nIn = linNIn lin
      rows = chunks nIn (linWeights lin)
  mapM
    ( \(wRow, b) -> do
        prods <- zipWithM mul wRow xs
        s <- foldl1M add prods
        add s b
    )
    (zip rows (linBias lin))

forwardMLP :: MLP -> [Value] -> IO [Value]
forwardMLP mlp = go (mlpLayers mlp)
  where
    go [] xs = return xs
    go [l] xs = forwardLinear l xs >>= mapM sigmoid
    go (l : ls) xs = forwardLinear l xs >>= mapM relu >>= go ls

parameters :: MLP -> [Value]
parameters = concatMap linearParams . mlpLayers

linearParams :: Linear -> [Value]
linearParams l = linWeights l ++ linBias l

zeroGrads :: MLP -> IO ()
zeroGrads = mapM_ zeroGrad . parameters

sgdStep :: Double -> MLP -> IO ()
sgdStep lr mlp = mapM_ updateParam (parameters mlp)
  where
    updateParam p = do
      d <- getData p
      g <- getGrad p
      setData p (d - lr * g)

chunks :: Int -> [a] -> [[a]]
chunks _ [] = []
chunks n xs = let (h, t) = splitAt n xs in h : chunks n t
