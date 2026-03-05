module Autograd
  ( Value,
    val,
    getData,
    getGrad,
    backward,
    relu,
    sigmoid,
    softmax,
    add,
    mul,
    sub,
    divide,
    power,
  )
where

import Data.IORef

data Value = Value
  { _data :: Double,
    _grad :: IORef Double,
    _backward :: IO ()
  }

val :: Double -> IO Value
val x = do
  g <- newIORef 0.0
  return $ Value x g (return ())

getData :: Value -> Double
getData = _data

getGrad :: Value -> IO Double
getGrad v = readIORef (_grad v)

addGrad :: Value -> Double -> IO ()
addGrad v d = modifyIORef' (_grad v) (+ d)

add :: Value -> Value -> IO Value
add a b = do
  g <- newIORef 0.0
  let out = Value (_data a + _data b) g bwd
      bwd = do
        og <- readIORef g
        addGrad a og
        addGrad b og
        _backward a
        _backward b
  return out

sub :: Value -> Value -> IO Value
sub a b = do
  g <- newIORef 0.0
  let out = Value (_data a - _data b) g bwd
      bwd = do
        og <- readIORef g
        addGrad a og
        addGrad b (negate og)
        _backward a
        _backward b
  return out

mul :: Value -> Value -> IO Value
mul a b = do
  g <- newIORef 0.0
  let out = Value (_data a * _data b) g bwd
      bwd = do
        og <- readIORef g
        addGrad a (og * _data b)
        addGrad b (og * _data a)
        _backward a
        _backward b
  return out

divide :: Value -> Value -> IO Value
divide a b = do
  g <- newIORef 0.0
  let out = Value (_data a / _data b) g bwd
      bwd = do
        og <- readIORef g
        addGrad a (og / _data b)
        addGrad b (negate og * _data a / (_data b * _data b))
        _backward a
        _backward b
  return out

power :: Value -> Double -> IO Value
power a e = do
  g <- newIORef 0.0
  let out = Value (_data a ** e) g bwd
      bwd = do
        og <- readIORef g
        addGrad a (og * e * (_data a ** (e - 1)))
        _backward a
  return out

relu :: Value -> IO Value
relu a = do
  g <- newIORef 0.0
  let y = max 0 (_data a)
      out = Value y g bwd
      bwd = do
        og <- readIORef g
        addGrad a (if _data a > 0 then og else 0)
        _backward a
  return out

sigmoid :: Value -> IO Value
sigmoid a = do
  g <- newIORef 0.0
  let s = 1 / (1 + exp (negate (_data a)))
      out = Value s g bwd
      bwd = do
        og <- readIORef g
        addGrad a (og * s * (1 - s))
        _backward a
  return out

softmax :: [Value] -> IO [Value]
softmax xs = do
  let maxX = maximum (map _data xs)
      exps_ = map (\x -> exp (_data x - maxX)) xs
      sumExps = sum exps_
      softmaxVals = map (/ sumExps) exps_
  outs <- mapM (\sv -> do g <- newIORef 0.0; return (sv, g)) softmaxVals
  let n = length xs
      outValues = zipWith (\(sv, g) x -> Value sv g (bwd sv g x outs)) outs xs
      bwd si gi xi allOuts = do
        og <- readIORef gi
        mapM_
          ( \(j, (sj, gj), xj) -> do
              let delta = if _data xi == _data xj && si == sj then 1 else 0
                  localGrad = og * si * (delta - sj)
              addGrad xj localGrad
          )
          (zip3 [0 :: Int ..] allOuts xs)
        _backward xi
  mapM_ return outValues
  return outValues

backward :: Value -> IO ()
backward v = do
  writeIORef (_grad v) 1.0
  _backward v

instance Num Value where
  (+) a b = error "Use monadic add: add a b"
  (*) a b = error "Use monadic mul: mul a b"
  (-) a b = error "Use monadic sub: sub a b"
  abs v = error "abs not supported"
  signum v = error "signum not supported"
  fromInteger n = error "fromInteger not supported; use val"
  negate v = error "Use monadic sub"
