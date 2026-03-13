module Tensor
  ( Tensor (..),
    fromList,
    fromList2D,
    toDoubles,
    tensorAdd,
    tensorSub,
    tensorMulElem,
    tensorRelu,
    tensorSigmoid,
    tensorTanh,
    matMul,
    transpose2D,
    addBias,
    tmap,
    tzipWith,
  )
where

import Autograd
import Control.Monad (zipWithM)

-- | A tensor is a flat list of Values with a shape descriptor
data Tensor = Tensor { tShape :: [Int], tElems :: [Value] }

-- | Smart constructors
fromList :: [Double] -> IO Tensor
fromList xs = do
  vs <- mapM val xs
  return $ Tensor [length xs] vs

fromList2D :: [[Double]] -> IO Tensor
fromList2D xss = do
  vs <- mapM val (concat xss)
  let rows = length xss
      cols = if null xss then 0 else length (head xss)
  return $ Tensor [rows, cols] vs

-- | Extract doubles from tensor (point-free)
toDoubles :: Tensor -> IO [Double]
toDoubles = mapM getData . tElems

-- | Higher-order combinator for element-wise binary ops
tzipWith :: (Value -> Value -> IO Value) -> Tensor -> Tensor -> IO Tensor
tzipWith f a b = do
  elems <- zipWithM f (tElems a) (tElems b)
  return $ Tensor (tShape a) elems

-- | Higher-order combinator for element-wise unary ops
tmap :: (Value -> IO Value) -> Tensor -> IO Tensor
tmap f t = do
  elems <- mapM f (tElems t)
  return $ Tensor (tShape t) elems

-- | Element-wise operations via tzipWith
tensorAdd :: Tensor -> Tensor -> IO Tensor
tensorAdd = tzipWith add

tensorSub :: Tensor -> Tensor -> IO Tensor
tensorSub = tzipWith sub

tensorMulElem :: Tensor -> Tensor -> IO Tensor
tensorMulElem = tzipWith mul

-- | Activation maps via tmap
tensorRelu :: Tensor -> IO Tensor
tensorRelu = tmap relu

tensorSigmoid :: Tensor -> IO Tensor
tensorSigmoid = tmap sigmoid

tensorTanh :: Tensor -> IO Tensor
tensorTanh = tmap tanhV

-- | Matrix multiplication: [m,k] x [k,n] -> [m,n]
matMul :: Tensor -> Tensor -> IO Tensor
matMul a b = case (tShape a, tShape b) of
  ([m, k], [k', n])
    | k /= k'  -> error $ "matMul: incompatible shapes " ++ show (tShape a) ++ " " ++ show (tShape b)
    | otherwise -> do
        let aRows = chunks k (tElems a)
            bT    = transpose2DList k n (tElems b)
        elems <- sequence
          [ dotProduct row col | row <- aRows, col <- bT ]
        return $ Tensor [m, n] elems
  _ -> error $ "matMul: expected 2D tensors, got shapes " ++ show (tShape a) ++ " " ++ show (tShape b)
  where
    dotProduct xs ys = do
      prods <- zipWithM mul xs ys
      foldl1M add prods

-- | Transpose a 2D tensor (pure index rearrangement)
transpose2D :: Tensor -> Tensor
transpose2D t = case tShape t of
  [r, c] -> Tensor [c, r] (concat $ transpose2DList r c (tElems t))
  s      -> error $ "transpose2D: expected 2D tensor, got shape " ++ show s

-- | Helper: transpose a flat list representing [r,c] into list of columns
transpose2DList :: Int -> Int -> [a] -> [[a]]
transpose2DList _r c elems =
  let rows = chunks c elems
  in  [ map (!! j) rows | j <- [0 .. c - 1] ]

-- | Split a list into chunks of size n
chunks :: Int -> [a] -> [[a]]
chunks _ [] = []
chunks n xs = let (h, t) = splitAt n xs in h : chunks n t

-- | Add a 1D bias tensor to each row of a 2D tensor
addBias :: Tensor -> Tensor -> IO Tensor
addBias t bias_ = case tShape t of
  [r, c] -> do
    let bElems = tElems bias_
        rows   = chunks c (tElems t)
    newElems <- concat <$> mapM (\row -> zipWithM add row bElems) rows
    return $ Tensor [r, c] newElems
  s -> error $ "addBias: expected 2D tensor, got shape " ++ show s
