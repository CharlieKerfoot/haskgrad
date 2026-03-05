module Main where

import Autograd
import Control.Monad (forM, forM_)
import Data.IORef
import Data.List (intercalate)

data Neuron = Neuron {weights :: [Value], bias :: Value}

data Layer = Layer {neurons :: [Neuron]}

data MLP = MLP {layers :: [Layer]}

newNeuron :: Int -> IO Neuron
newNeuron nIn = do
  ws <- mapM (\_ -> val (randomish nIn)) [1 .. nIn]
  b <- val 0.0
  return $ Neuron ws b
  where
    randomish n = 0.5 - fromIntegral (n `mod` 7) * 0.1

newLayer :: Int -> Int -> IO Layer
newLayer nIn nOut = do
  ns <- mapM (\i -> newNeuron nIn) [1 .. nOut]
  return $ Layer ns

newMLP :: [Int] -> IO MLP
newMLP sizes = do
  let pairs = zip sizes (tail sizes)
  ls <- mapM (uncurry newLayer) pairs
  return $ MLP ls

forwardNeuron :: Neuron -> [Value] -> IO Value
forwardNeuron n xs = do
  prods <- mapM (\(w, x) -> mul w x) (zip (weights n) xs)
  total <- foldl1M add prods
  withBias <- add total (bias n)
  sigmoid withBias

foldl1M :: (Monad m) => (a -> a -> m a) -> [a] -> m a
foldl1M _ [x] = return x
foldl1M f (x : xs) = do
  acc <- f x (head xs)
  foldl1M f (acc : tail xs)
foldl1M _ [] = error "empty list"

forwardLayer :: Layer -> [Value] -> Bool -> IO [Value]
forwardLayer l xs isLast =
  mapM
    ( \n -> do
        prods <- mapM (\(w, x) -> mul w x) (zip (weights n) xs)
        total <- foldl1M add prods
        withBias <- add total (bias n)
        if isLast then sigmoid withBias else relu withBias
    )
    (neurons l)

forwardMLP :: MLP -> [Value] -> IO [Value]
forwardMLP mlp xs = do
  let ls = layers mlp
      n = length ls
  foldM_layers xs (zip [1 ..] ls) n
  where
    foldM_layers cur [] _ = return cur
    foldM_layers cur ((i, l) : rest) total = do
      next <- forwardLayer l cur (i == total)
      foldM_layers next rest total

mseLoss :: [Value] -> [Double] -> IO Value
mseLoss preds targets = do
  diffs <- mapM (\(p, t) -> do tv <- val t; sub p tv) (zip preds targets)
  squares <- mapM (\d -> power d 2) diffs
  foldl1M add squares

zeroGrads :: MLP -> IO ()
zeroGrads mlp =
  forM_ (layers mlp) $ \layer ->
    forM_ (neurons layer) $ \neuron -> do
      forM_ (weights neuron) $ \w -> writeIORef (_grad w) 0.0
      writeIORef (_grad (bias neuron)) 0.0

updateParams :: MLP -> Double -> IO ()
updateParams mlp lr =
  forM_ (layers mlp) $ \layer ->
    forM_ (neurons layer) $ \neuron -> do
      forM_ (weights neuron) $ \w -> do
        g <- getGrad w
        let newD = getData w - lr * g
        return ()
      return ()

getAllParams :: MLP -> [Value]
getAllParams mlp =
  concatMap
    (\layer -> concatMap (\n -> weights n ++ [bias n]) (neurons layer))
    (layers mlp)

updateParamsList :: [Value] -> Double -> IO [Value]
updateParamsList params lr = do
  mapM
    ( \p -> do
        g <- getGrad p
        val (getData p - lr * g)
    )
    params

xorData :: [([Double], Double)]
xorData =
  [ ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
  ]

trainStep :: MLP -> Double -> IO (MLP, Double)
trainStep mlp lr = do
  totalLoss <- newIORef 0.0
  forM_ xorData $ \(xs, target) -> do
    zeroGrads mlp
    inputs <- mapM val xs
    [pred_] <- forwardMLP mlp inputs
    loss <- mseLoss [pred_] [target]
    modifyIORef totalLoss (+ getData loss)
    backward loss
  l <- readIORef totalLoss
  return (mlp, l / fromIntegral (length xorData))

main :: IO ()
main = do
  mlp <- newMLP [2, 4, 1]
  putStrLn "Training XOR MLP (scalar autograd)\n"
  putStrLn $ padR 8 "Epoch" ++ padR 12 "Loss" ++ "Predictions"
  putStrLn $ replicate 60 '-'
  forM_ [1 .. 200 :: Int] $ \epoch -> do
    (_, loss) <- trainStep mlp 0.1
    if epoch `mod` 20 == 0
      then do
        preds <- forM xorData $ \(xs, _) -> do
          inputs <- mapM val xs
          [p] <- forwardMLP mlp inputs
          return (getData p)
        let predStr = intercalate "  " (map (take 5 . show) preds)
        putStrLn $ padR 8 (show epoch) ++ padR 12 (take 7 (show loss)) ++ predStr
      else return ()
  putStrLn "\nFinal predictions (expect: ~0, ~1, ~1, ~0):"
  forM_ xorData $ \(xs, target) -> do
    inputs <- mapM val xs
    [p] <- forwardMLP mlp inputs
    putStrLn $
      "  Input: "
        ++ show (map round xs :: [Int])
        ++ "  Target: "
        ++ show (round target :: Int)
        ++ "  Pred: "
        ++ take 6 (show (getData p))

padR :: Int -> String -> String
padR n s = s ++ replicate (max 0 (n - length s)) ' '
