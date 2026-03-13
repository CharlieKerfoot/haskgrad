module Main where

import Autograd (add, backward, foldl1M, getData, power, sub, val)
import Control.Monad (forM_, when)
import NN (forwardMLP, mkRNG, newMLP, sgdStep, zeroGrads)
import Numeric (showFFloat)

xorData :: [([Double], Double)]
xorData =
  [ ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
  ]

main :: IO ()
main = do
  rng <- mkRNG 42
  mlp <- newMLP rng [2, 8, 1]

  putStrLn "Training XOR MLP (scalar autograd)\n"
  putStrLn $ padR 8 "Epoch" ++ padR 14 "Loss" ++ "Predictions"
  putStrLn $ replicate 60 '-'

  let epochs = 1000 :: Int
      lr = 0.05

  forM_ [1 .. epochs] $ \epoch -> do
    zeroGrads mlp

    losses <-
      mapM
        ( \(xs, target) -> do
            inputs <- mapM val xs
            [p] <- forwardMLP mlp inputs
            tv <- val target
            diff <- sub p tv
            power diff 2
        )
        xorData

    loss <- foldl1M add losses

    backward loss
    sgdStep lr mlp

    when (epoch `mod` 100 == 0 || epoch == 1) $ do
      lossVal <- getData loss
      preds <-
        mapM
          ( \(xs, _) -> do
              inputs <- mapM val xs
              [p] <- forwardMLP mlp inputs
              getData p
          )
          xorData
      let fmtD = showFFloat (Just 4)
          predStr = unwords $ map (\p -> padR 8 (fmtD p "")) preds
      putStrLn $ padR 8 (show epoch) ++ padR 14 (fmtD lossVal "") ++ predStr

  putStrLn "\nFinal predictions (expect: ~0, ~1, ~1, ~0):"
  forM_ xorData $ \(xs, target) -> do
    inputs <- mapM val xs
    [p] <- forwardMLP mlp inputs
    pVal <- getData p
    putStrLn $
      "  Input: "
        ++ show (map round xs :: [Int])
        ++ "  Target: "
        ++ show (round target :: Int)
        ++ "  Pred: "
        ++ showFFloat (Just 4) pVal ""

padR :: Int -> String -> String
padR n s = s ++ replicate (max 0 (n - length s)) ' '
